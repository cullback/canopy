use std::collections::HashMap;

use rand::Rng;
use rand::seq::IndexedRandom;

use crate::game::{Status, StochasticGame};
use crate::player::Player;

/// Per-edge statistics stored inside a node.
#[derive(Clone, Debug)]
struct EdgeStats {
    visits: u32,
    total_value: f32,
}

impl EdgeStats {
    fn new() -> Self {
        Self {
            visits: 0,
            total_value: 0.0,
        }
    }
}

/// Children of a node — either player actions or chance outcomes.
#[derive(Clone, Debug)]
enum Children {
    Actions(HashMap<usize, (EdgeStats, Option<usize>)>),
    Chances(HashMap<usize, (EdgeStats, Option<usize>)>),
}

/// A single MCTS node.
#[derive(Clone, Debug)]
struct Node {
    visits: u32,
    player: Player,
    children: Children,
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
pub struct Mcts {
    nodes: Vec<Node>,
    config: MctsConfig,
    action_buf: Vec<usize>,
    chance_buf: Vec<(usize, f32)>,
}

impl Mcts {
    pub fn new(config: MctsConfig) -> Self {
        Self {
            nodes: Vec::new(),
            config,
            action_buf: Vec::new(),
            chance_buf: Vec::new(),
        }
    }

    /// Run MCTS from the given root state and return a visit-count distribution
    /// over legal actions.
    pub fn search<G: StochasticGame>(&mut self, root: &G, rng: &mut impl Rng) -> Vec<(usize, u32)> {
        self.nodes.clear();
        self.create_node(root);

        for _ in 0..self.config.simulations {
            let mut state = root.clone();
            self.simulate(0, &mut state, rng);
        }

        let root_node = &self.nodes[0];
        let Children::Actions(ref actions) = root_node.children else {
            panic!("search called on a chance node root");
        };
        let mut action_visits: Vec<(usize, u32)> = actions
            .iter()
            .map(|(&a, (stats, _))| (a, stats.visits))
            .collect();
        action_visits.sort_by(|a, b| b.1.cmp(&a.1));
        action_visits
    }

    /// Pick the best action by visit count.
    pub fn best_action<G: StochasticGame>(
        &mut self,
        root: &G,
        rng: &mut impl Rng,
    ) -> Option<usize> {
        let visits = self.search(root, rng);
        visits.into_iter().next().map(|(a, _)| a)
    }

    fn create_node<G: StochasticGame>(&mut self, state: &G) -> usize {
        let Status::Ongoing(player) = state.status() else {
            panic!("create_node called on terminal state");
        };

        let children = if state.is_chance_node() {
            self.chance_buf.clear();
            state.chance_outcomes(&mut self.chance_buf);
            Children::Chances(
                self.chance_buf
                    .iter()
                    .map(|&(o, _)| (o, (EdgeStats::new(), None)))
                    .collect(),
            )
        } else {
            self.action_buf.clear();
            state.legal_actions(&mut self.action_buf);
            Children::Actions(
                self.action_buf
                    .iter()
                    .map(|&a| (a, (EdgeStats::new(), None)))
                    .collect(),
            )
        };

        let idx = self.nodes.len();
        self.nodes.push(Node {
            visits: 0,
            player,
            children,
        });
        idx
    }

    /// Selection + expansion + rollout. Returns reward from P1's perspective.
    fn simulate<G: StochasticGame>(
        &mut self,
        node_idx: usize,
        state: &mut G,
        rng: &mut impl Rng,
    ) -> f32 {
        if let Status::Terminal(reward) = state.status() {
            return reward;
        }

        let player = self.nodes[node_idx].player;
        match &self.nodes[node_idx].children {
            Children::Chances(_) => self.simulate_chance(node_idx, player, state, rng),
            Children::Actions(_) => self.simulate_action(node_idx, player, state, rng),
        }
    }

    fn simulate_chance<G: StochasticGame>(
        &mut self,
        node_idx: usize,
        player: Player,
        state: &mut G,
        rng: &mut impl Rng,
    ) -> f32 {
        self.chance_buf.clear();
        state.chance_outcomes(&mut self.chance_buf);
        let outcome = sample_weighted(&self.chance_buf, rng);

        state.apply_chance(outcome);

        let Children::Chances(ref chances) = self.nodes[node_idx].children else {
            unreachable!();
        };
        let (edge, child_idx) = chances.get(&outcome).map(|(e, c)| (e.clone(), *c)).unwrap();

        let reward = if let Some(ci) = child_idx {
            self.simulate(ci, state, rng)
        } else if edge.visits == 0 || matches!(state.status(), Status::Terminal(_)) {
            self.rollout(state, rng)
        } else {
            let new_idx = self.create_node(state);
            let Children::Chances(ref mut chances) = self.nodes[node_idx].children else {
                unreachable!();
            };
            chances.get_mut(&outcome).unwrap().1 = Some(new_idx);
            self.simulate(new_idx, state, rng)
        };

        let Children::Chances(ref mut chances) = self.nodes[node_idx].children else {
            unreachable!();
        };
        let edge = &mut chances.get_mut(&outcome).unwrap().0;
        edge.visits += 1;
        edge.total_value += reward_for(reward, player);
        self.nodes[node_idx].visits += 1;

        reward
    }

    fn simulate_action<G: StochasticGame>(
        &mut self,
        node_idx: usize,
        player: Player,
        state: &mut G,
        rng: &mut impl Rng,
    ) -> f32 {
        let action = self.select_ucb(node_idx);

        state.apply_action(action);

        let Children::Actions(ref actions) = self.nodes[node_idx].children else {
            unreachable!();
        };
        let (edge, child_idx) = actions.get(&action).map(|(e, c)| (e.clone(), *c)).unwrap();

        let reward = if let Some(ci) = child_idx {
            self.simulate(ci, state, rng)
        } else if edge.visits == 0 || matches!(state.status(), Status::Terminal(_)) {
            self.rollout(state, rng)
        } else {
            let new_idx = self.create_node(state);
            let Children::Actions(ref mut actions) = self.nodes[node_idx].children else {
                unreachable!();
            };
            actions.get_mut(&action).unwrap().1 = Some(new_idx);
            self.simulate(new_idx, state, rng)
        };

        let Children::Actions(ref mut actions) = self.nodes[node_idx].children else {
            unreachable!();
        };
        let edge = &mut actions.get_mut(&action).unwrap().0;
        edge.visits += 1;
        edge.total_value += reward_for(reward, player);
        self.nodes[node_idx].visits += 1;

        reward
    }

    fn select_ucb(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        let Children::Actions(ref actions) = node.children else {
            panic!("select_ucb called on chance node");
        };
        let ln_parent = (node.visits.max(1) as f32).ln();

        let mut best_action = None;
        let mut best_score = f32::NEG_INFINITY;

        for (&action, (stats, _)) in actions {
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

    fn rollout<G: StochasticGame>(&mut self, state: &mut G, rng: &mut impl Rng) -> f32 {
        loop {
            if let Status::Terminal(reward) = state.status() {
                return reward;
            }
            if state.is_chance_node() {
                self.chance_buf.clear();
                state.chance_outcomes(&mut self.chance_buf);
                let outcome = sample_weighted(&self.chance_buf, rng);
                state.apply_chance(outcome);
            } else {
                self.action_buf.clear();
                state.legal_actions(&mut self.action_buf);
                let action = *self.action_buf.choose(rng).unwrap();
                state.apply_action(action);
            }
        }
    }
}

/// Convert a P1-perspective reward to the value for `player` (zero-sum).
fn reward_for(p1_reward: f32, player: Player) -> f32 {
    match player {
        Player::One => p1_reward,
        Player::Two => -p1_reward,
    }
}

fn sample_weighted(items: &[(usize, f32)], rng: &mut impl Rng) -> usize {
    let total: f32 = items.iter().map(|(_, p)| p).sum();
    let mut r = rng.random_range(0.0..total);
    for &(item, p) in items {
        r -= p;
        if r <= 0.0 {
            return item;
        }
    }
    items.last().unwrap().0
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

    impl StochasticGame for TrivialGame {
        fn is_chance_node(&self) -> bool {
            false
        }
        fn chance_outcomes(&self, _buf: &mut Vec<(usize, f32)>) {}
        fn apply_chance(&mut self, _outcome: usize) {}
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
