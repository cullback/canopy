use std::future::Future;

use crate::eval::Evaluation;
use crate::game::{Game, Status};
use crate::mcts::{Search, Step};
use crate::nn::StateEncoder;

use super::Sample;

/// Per-game config knobs derived from [`super::TrainConfig`].
pub(super) struct ActorConfig {
    pub explore_moves: u32,
    pub playout_cap_full_prob: f32,
    pub playout_cap_fast_sims: u32,
    pub effective_sims: u32,
}

/// Result of a single self-play game.
pub(super) struct GameResult {
    pub samples: Vec<Sample>,
    pub reward: f32,
}

/// Play one full self-play game, returning collected training samples.
///
/// Pure game logic: chance nodes → forced moves → MCTS search → sample → apply.
/// Leaf evaluation is delegated to the caller-provided `infer` closure, keeping
/// this function free of any channel, batching, or GPU knowledge.
pub(super) async fn play_game<G, E, F, Fut>(
    search: &mut Search<G>,
    actor_config: &ActorConfig,
    infer: F,
    rng: &mut fastrand::Rng,
) -> GameResult
where
    G: Game,
    E: StateEncoder<G>,
    F: Fn(Vec<f32>) -> Fut,
    Fut: Future<Output = (Vec<f32>, f32)>,
{
    let mut samples: Vec<Sample> = Vec::new();
    let mut turn_count: u32 = 0;
    let mut last_sign: Option<f32> = None;
    let mut actions_buf = Vec::new();
    let mut encode_buf = Vec::with_capacity(E::FEATURE_SIZE);

    loop {
        // Resolve chance nodes
        if let Some(action) = search.state().sample_chance(rng) {
            search.apply_action(action);
            continue;
        }

        // Terminal check
        if let Status::Terminal(reward) = search.state().status() {
            for s in &mut samples {
                s.z *= reward;
                s.game_length = turn_count;
            }
            return GameResult { samples, reward };
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
            search.apply_action(actions_buf[0]);
            continue;
        }

        // Playout cap randomization
        let is_full_search = rng.f32() < actor_config.playout_cap_full_prob;
        search.set_num_simulations(if is_full_search {
            actor_config.effective_sims
        } else {
            actor_config.playout_cap_fast_sims
        });

        // Encode features before search (for training sample)
        let mut features_buf = Vec::with_capacity(E::FEATURE_SIZE);
        E::encode(search.state(), &mut features_buf);

        // MCTS search with eval requests sent via infer closure
        let mut evals: Vec<Evaluation> = vec![];
        let result = loop {
            match search.step(&evals, rng) {
                Step::NeedsEval(states) => {
                    evals.clear();
                    for pending_state in states {
                        let sign = match pending_state.status() {
                            Status::Ongoing => pending_state.current_sign(),
                            Status::Terminal(_) => 1.0,
                        };
                        encode_buf.clear();
                        E::encode(pending_state, &mut encode_buf);
                        let (policy_logits, value) = infer(encode_buf.clone()).await;
                        evals.push(Evaluation {
                            policy_logits,
                            value: value * sign,
                        });
                    }
                }
                Step::Done(result) => break result,
            }
        };

        // Create training sample
        let q = result.value * sign;
        let chosen = if turn_count <= actor_config.explore_moves {
            sample_from_policy(&result.policy, rng)
        } else {
            result.selected_action
        };

        let value_correction = (result.value - result.network_value).abs();
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

        samples.push(Sample {
            features: features_buf.into_boxed_slice(),
            policy_target: result.policy.into_boxed_slice(),
            z: sign,
            q,
            full_search: is_full_search,
            move_number: turn_count,
            game_length: 0, // backfilled at game end
            network_value: result.network_value * sign,
            value_correction,
            q_std,
            prior_agrees,
        });

        search.apply_action(chosen);
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
