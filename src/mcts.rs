use std::collections::HashMap;
use std::hash::Hash;

use rand::Rng;
use rand::seq::IndexedRandom;

use crate::game::{Game, StochasticGame};

/// Per-edge statistics stored inside a node.
#[derive(Clone, Debug)]
struct EdgeStats {
    visits: u32,
    total_value: f32,
}

/// A single MCTS node.
#[derive(Clone, Debug)]
struct Node<A: Copy + Eq + Hash, P: Copy + Eq + Hash> {
    visits: u32,
    player: P,
    is_chance: bool,
    /// Maps action -> (edge stats, optional child node index).
    children: HashMap<A, (EdgeStats, Option<usize>)>,
}

/// Configuration for MCTS search.
pub struct MctsConfig {
    /// UCB exploration constant. sqrt(2) is a common default.
    pub exploration: f32,
    /// Number of simulations to run.
    pub simulations: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            exploration: std::f32::consts::SQRT_2,
            simulations: 1000,
        }
    }
}

/// Vanilla UCT Monte Carlo Tree Search.
pub struct Mcts<G: Game> {
    nodes: Vec<Node<G::Action, G::Player>>,
    config: MctsConfig,
}

impl<G: StochasticGame> Mcts<G> {
    pub fn new(config: MctsConfig) -> Self {
        Self {
            nodes: Vec::new(),
            config,
        }
    }

    /// Run MCTS from the given root state and return a visit-count distribution
    /// over legal actions.
    pub fn search(&mut self, root: &G, rng: &mut impl Rng) -> Vec<(G::Action, u32)> {
        self.nodes.clear();
        self.create_node(root);

        for _ in 0..self.config.simulations {
            let mut state = root.clone();
            let rewards = self.simulate(0, &mut state, rng);
            self.backprop(0, &rewards);
        }

        let root_node = &self.nodes[0];
        let mut action_visits: Vec<(G::Action, u32)> = root_node
            .children
            .iter()
            .map(|(&a, (stats, _))| (a, stats.visits))
            .collect();
        action_visits.sort_by(|a, b| b.1.cmp(&a.1));
        action_visits
    }

    /// Pick the best action by visit count.
    pub fn best_action(&mut self, root: &G, rng: &mut impl Rng) -> Option<G::Action> {
        let visits = self.search(root, rng);
        visits.into_iter().next().map(|(a, _)| a)
    }

    fn create_node(&mut self, state: &G) -> usize {
        let is_chance = state.is_chance_node();
        let player = state.current_player();

        let children = if is_chance {
            state
                .chance_outcomes()
                .into_iter()
                .map(|(a, _prob)| {
                    (
                        a,
                        (
                            EdgeStats {
                                visits: 0,
                                total_value: 0.0,
                            },
                            None,
                        ),
                    )
                })
                .collect()
        } else {
            state
                .legal_actions()
                .into_iter()
                .map(|a| {
                    (
                        a,
                        (
                            EdgeStats {
                                visits: 0,
                                total_value: 0.0,
                            },
                            None,
                        ),
                    )
                })
                .collect()
        };

        let idx = self.nodes.len();
        self.nodes.push(Node {
            visits: 0,
            player,
            is_chance,
            children,
        });
        idx
    }

    /// Selection + expansion + rollout. Returns rewards from the perspective of
    /// each player.
    fn simulate(
        &mut self,
        node_idx: usize,
        state: &mut G,
        rng: &mut impl Rng,
    ) -> HashMap<G::Player, f32> {
        if state.is_terminal() {
            return state.rewards();
        }

        let action = if self.nodes[node_idx].is_chance {
            // Sample from chance distribution.
            let outcomes = state.chance_outcomes();
            let total: f32 = outcomes.iter().map(|(_, p)| p).sum();
            let mut r = rng.random_range(0.0..total);
            let mut chosen = outcomes[0].0;
            for (a, p) in &outcomes {
                r -= p;
                if r <= 0.0 {
                    chosen = *a;
                    break;
                }
            }
            chosen
        } else {
            self.select_ucb(node_idx)
        };

        state.apply_action(action);

        let (edge, child_idx) = self.nodes[node_idx]
            .children
            .get(&action)
            .map(|(e, c)| (e.clone(), *c))
            .unwrap();

        let rewards = if let Some(ci) = child_idx {
            // Already expanded — recurse.
            self.simulate(ci, state, rng)
        } else if edge.visits == 0 {
            // First visit — rollout.
            self.rollout(state, rng)
        } else {
            // Second visit — expand.
            let new_idx = self.create_node(state);
            self.nodes[node_idx].children.get_mut(&action).unwrap().1 = Some(new_idx);
            self.simulate(new_idx, state, rng)
        };

        // Update edge stats.
        let player = self.nodes[node_idx].player;
        let edge = &mut self.nodes[node_idx].children.get_mut(&action).unwrap().0;
        edge.visits += 1;
        edge.total_value += rewards.get(&player).copied().unwrap_or(0.0);

        self.nodes[node_idx].visits += 1;

        rewards
    }

    fn select_ucb(&self, node_idx: usize) -> G::Action {
        let node = &self.nodes[node_idx];
        let ln_parent = (node.visits.max(1) as f32).ln();

        let mut best_action = None;
        let mut best_score = f32::NEG_INFINITY;

        for (&action, (stats, _)) in &node.children {
            let score = if stats.visits == 0 {
                f32::INFINITY
            } else {
                let exploit = stats.total_value / stats.visits as f32;
                let explore = self.config.exploration * (ln_parent / stats.visits as f32).sqrt();
                exploit + explore
            };
            if score > best_score {
                best_score = score;
                best_action = Some(action);
            }
        }

        best_action.expect("node has no children")
    }

    fn rollout(&self, state: &mut G, rng: &mut impl Rng) -> HashMap<G::Player, f32> {
        while !state.is_terminal() {
            if state.is_chance_node() {
                let outcomes = state.chance_outcomes();
                let total: f32 = outcomes.iter().map(|(_, p)| p).sum();
                let mut r = rng.random_range(0.0..total);
                for (a, p) in &outcomes {
                    r -= p;
                    if r <= 0.0 {
                        state.apply_action(*a);
                        break;
                    }
                }
            } else {
                let actions = state.legal_actions();
                let action = *actions.choose(rng).unwrap();
                state.apply_action(action);
            }
        }
        state.rewards()
    }

    fn backprop(&mut self, _node_idx: usize, _rewards: &HashMap<G::Player, f32>) {
        // Stats are updated during simulate's unwind — nothing extra needed.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, StochasticGame};

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
        type Action = u8;
        type Player = u8;

        fn current_player(&self) -> u8 {
            0
        }
        fn legal_actions(&self) -> Vec<u8> {
            if self.done { vec![] } else { vec![0, 1] }
        }
        fn apply_action(&mut self, action: u8) {
            self.chose_win = action == 0;
            self.done = true;
        }
        fn is_terminal(&self) -> bool {
            self.done
        }
        fn rewards(&self) -> HashMap<u8, f32> {
            let mut m = HashMap::new();
            m.insert(0, if self.chose_win { 1.0 } else { 0.0 });
            m
        }
        fn state_key(&self) -> u64 {
            self.done as u64 * 2 + self.chose_win as u64
        }
        fn action_index(action: &u8) -> usize {
            *action as usize
        }
        fn action_space_size() -> usize {
            2
        }
    }

    impl StochasticGame for TrivialGame {
        fn is_chance_node(&self) -> bool {
            false
        }
        fn chance_outcomes(&self) -> Vec<(u8, f32)> {
            vec![]
        }
    }

    #[test]
    fn mcts_finds_winning_action() {
        let game = TrivialGame::new();
        let config = MctsConfig {
            simulations: 500,
            ..Default::default()
        };
        let mut mcts = Mcts::new(config);
        let mut rng = rand::rng();
        let best = mcts.best_action(&game, &mut rng);
        assert_eq!(best, Some(0), "MCTS should find that action 0 wins");
    }
}
