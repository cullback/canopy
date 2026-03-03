use crate::eval::Evaluator;
use crate::game::{Game, Status};
use crate::mcts::{Config, Search, Step};

struct Slot<G: Game> {
    state: G,
    search: Option<Search<G>>,
    pending_step: Option<Step<G>>,
    chance_buf: Vec<(usize, f32)>,
}

/// Run `num_games` self-play games with batched evaluation.
///
/// Games run in concurrent slots (up to `batch_size` at once). Each slot uses
/// the MCTS state machine to pause at `NeedsEval`, allowing the orchestrator
/// to collect pending evaluations into a batch before calling the evaluator.
///
/// Returns the terminal reward (P1 perspective) for each completed game.
pub fn batched_self_play<G: Game, E: Evaluator<G> + ?Sized>(
    game: &G,
    evaluator: &E,
    config: &Config,
    num_games: u32,
    batch_size: u32,
    rng: &mut fastrand::Rng,
) -> Vec<f32> {
    let active = batch_size.min(num_games) as usize;
    let mut slots: Vec<Slot<G>> = (0..active)
        .map(|_| Slot {
            state: game.clone(),
            search: None,
            pending_step: None,
            chance_buf: Vec::new(),
        })
        .collect();
    let mut games_started = active as u32;
    let mut results = Vec::with_capacity(num_games as usize);

    // Scratch space for batch collection
    let mut batch_indices = Vec::new();
    let mut batch_states: Vec<G> = Vec::new();

    while results.len() < num_games as usize {
        // Phase A — Advance non-search states
        for slot in slots.iter_mut() {
            loop {
                // Terminal check
                if let Status::Terminal(reward) = slot.state.status() {
                    results.push(reward);
                    // Recycle slot if games remain
                    if games_started < num_games {
                        slot.state = game.clone();
                        slot.search = None;
                        slot.pending_step = None;
                        games_started += 1;
                    } else {
                        slot.search = None;
                        slot.pending_step = None;
                        break;
                    }
                    // Continue the loop to check the fresh game (it shouldn't be terminal,
                    // but handle it uniformly)
                    continue;
                }

                // Chance node — sample and apply
                slot.chance_buf.clear();
                slot.state.chance_outcomes(&mut slot.chance_buf);
                if !slot.chance_buf.is_empty() {
                    let action = sample_chance(&slot.chance_buf, rng);
                    slot.state.apply_action(action);
                    if let Some(search) = &mut slot.search {
                        search.advance(action);
                    }
                    continue;
                }

                // Decision node — resume existing search or start fresh
                if let Some(search) = &mut slot.search {
                    let step = search.resume(&slot.state, config, rng);
                    slot.pending_step = Some(step);
                } else {
                    let (search, step) = Search::new(&slot.state, config, rng);
                    slot.search = Some(search);
                    slot.pending_step = Some(step);
                }

                break;
            }
        }

        // Phase B — Collect NeedsEval batch
        batch_indices.clear();
        batch_states.clear();
        for (i, slot) in slots.iter_mut().enumerate() {
            if let Some(Step::NeedsEval(_)) = &slot.pending_step
                && let Some(Step::NeedsEval(state)) = slot.pending_step.take()
            {
                batch_indices.push(i);
                batch_states.push(state);
            }
        }

        if batch_indices.is_empty() {
            // All slots are either terminal with no recycling or Done — process Done steps
            for slot in slots.iter_mut() {
                if let Some(Step::Done(_)) = &slot.pending_step
                    && let Some(Step::Done(result)) = slot.pending_step.take()
                {
                    let action = greedy_action(&result.policy);
                    slot.state.apply_action(action);
                    if let Some(search) = &mut slot.search {
                        search.advance(action);
                    }
                }
            }
            continue;
        }

        // Phase C — Evaluate batch
        let outputs: Vec<_> = batch_states
            .iter()
            .map(|state| evaluator.evaluate(state, rng))
            .collect();

        // Phase D — Supply results back
        for (idx, output) in batch_indices.iter().zip(outputs) {
            let slot = &mut slots[*idx];
            let search = slot.search.as_mut().unwrap();
            let step = search.supply(output, config, rng);
            match step {
                Step::Done(result) => {
                    let action = greedy_action(&result.policy);
                    slot.state.apply_action(action);
                    if let Some(search) = &mut slot.search {
                        search.advance(action);
                    }
                    slot.pending_step = None;
                }
                needs_eval @ Step::NeedsEval(_) => {
                    slot.pending_step = Some(needs_eval);
                }
            }
        }
    }

    results
}

fn greedy_action(policy: &[f32]) -> usize {
    policy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn sample_chance(outcomes: &[(usize, f32)], rng: &mut fastrand::Rng) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::RolloutEvaluator;
    use crate::game::Status;
    use crate::mcts::Config;
    use crate::player::Player;

    /// Trivial one-step game: player picks action 0 (win) or 1 (lose).
    #[derive(Clone)]
    struct TrivialGame {
        done: bool,
        chose_win: bool,
    }

    impl TrivialGame {
        fn new() -> Self {
            Self {
                done: false,
                chose_win: false,
            }
        }
    }

    impl Game for TrivialGame {
        const NUM_ACTIONS: usize = 2;

        fn status(&self) -> Status {
            if self.done {
                Status::Terminal(if self.chose_win { 1.0 } else { -1.0 })
            } else {
                Status::Ongoing(Player::One)
            }
        }
        fn legal_actions(&self, buf: &mut Vec<usize>) {
            if !self.done {
                buf.push(0);
                buf.push(1);
            }
        }
        fn apply_action(&mut self, action: usize) {
            self.chose_win = action == 0;
            self.done = true;
        }
    }

    #[test]
    fn batched_self_play_completes() {
        let game = TrivialGame::new();
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
        let config = Config {
            num_simulations: 50,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let results = batched_self_play(&game, &evaluator, &config, 8, 4, &mut rng);

        assert_eq!(results.len(), 8, "should produce exactly 8 results");
        for &r in &results {
            assert!(
                r == 1.0 || r == -1.0,
                "TrivialGame reward should be ±1.0, got {r}"
            );
        }
    }
}
