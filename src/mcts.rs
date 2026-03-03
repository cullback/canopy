use std::collections::HashMap;

use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::eval::Evaluator;
use crate::game::{Game, Status};

// ── Public types ──────────────────────────────────────────────────────

/// MCTS search configuration.
pub struct Config {
    pub num_simulations: u32,
    pub cpuct: f32,
    pub fpu_reduction: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            cpuct: 2.5,
            fpu_reduction: 0.0,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }
}

/// Result of an MCTS search.
pub struct SearchResult {
    /// Visit-count policy over `[0, NUM_ACTIONS)`.
    pub policy: Vec<f32>,
    /// Root value estimate from P1's perspective.
    pub value: f32,
}

// ── Internal types ────────────────────────────────────────────────────

type NodeId = usize;

struct Edge {
    action: usize,
    child: Option<NodeId>,
    prior: f32,
    visits: u32,
    total_value: f32,
}

struct Node {
    status: Status,
    is_chance: bool,
    edges: Vec<Edge>,
}

#[derive(Default)]
struct Tree {
    nodes: Vec<Node>,
    table: HashMap<u64, NodeId>,
}

/// Scratch buffers reused across simulations.
#[derive(Default)]
struct Bufs {
    actions: Vec<usize>,
    chances: Vec<(usize, f32)>,
    path: Vec<(NodeId, usize)>,
}

// ── Public API ────────────────────────────────────────────────────────

/// Run MCTS from the given root state and return a policy + value.
pub fn search<G: Game>(
    root_state: &G,
    evaluator: &impl Evaluator<G>,
    config: &Config,
    rng: &mut impl Rng,
) -> SearchResult {
    if let Status::Terminal(reward) = root_state.status() {
        return SearchResult {
            policy: vec![0.0; G::NUM_ACTIONS],
            value: reward,
        };
    }

    let mut tree = Tree::default();
    let mut bufs = Bufs::default();

    let (root, _) = expand(&mut tree, root_state, evaluator, &mut bufs);

    // Dirichlet noise on root priors (skip for chance roots)
    if !tree.nodes[root].is_chance {
        add_dirichlet_noise(
            &mut tree.nodes[root].edges,
            config.dirichlet_alpha,
            config.dirichlet_epsilon,
            rng,
        );
    }

    for _ in 0..config.num_simulations {
        simulate(
            &mut tree, root, root_state, evaluator, config, rng, &mut bufs,
        );
    }

    // Extract policy from root visit counts
    let root_node = &tree.nodes[root];
    let total_visits: u32 = root_node.edges.iter().map(|e| e.visits).sum();
    let mut policy = vec![0.0f32; G::NUM_ACTIONS];
    if total_visits > 0 {
        for edge in &root_node.edges {
            policy[edge.action] = edge.visits as f32 / total_visits as f32;
        }
    }

    SearchResult {
        policy,
        value: node_value(root_node),
    }
}

// ── Tree operations ───────────────────────────────────────────────────

fn simulate<G: Game>(
    tree: &mut Tree,
    root: NodeId,
    root_state: &G,
    evaluator: &impl Evaluator<G>,
    config: &Config,
    rng: &mut impl Rng,
    bufs: &mut Bufs,
) {
    bufs.path.clear();
    let mut current = root;
    let mut state = root_state.clone();

    let value = loop {
        let node = &tree.nodes[current];

        if let Status::Terminal(reward) = node.status {
            break reward;
        }

        let edge_idx = if node.is_chance {
            sample_chance_edge(node, rng)
        } else {
            select_edge(node, config)
        };

        bufs.path.push((current, edge_idx));

        let edge = &tree.nodes[current].edges[edge_idx];
        let action = edge.action;
        let has_child = edge.child;

        state.apply_action(action);

        match has_child {
            Some(child) => current = child,
            None => {
                let (child_id, val) = expand(tree, &state, evaluator, bufs);
                let &(nid, eidx) = bufs.path.last().unwrap();
                tree.nodes[nid].edges[eidx].child = Some(child_id);
                if let Some(v) = val {
                    break v;
                }
                // Chance node — no eval, keep descending
                current = child_id;
            }
        }
    };

    // Backprop — value is P1-perspective, no sign flipping
    for &(nid, eidx) in bufs.path.iter().rev() {
        let edge = &mut tree.nodes[nid].edges[eidx];
        edge.visits += 1;
        edge.total_value += value;
    }
}

fn expand<G: Game>(
    tree: &mut Tree,
    state: &G,
    evaluator: &impl Evaluator<G>,
    bufs: &mut Bufs,
) -> (NodeId, Option<f32>) {
    // Transposition hit — reuse existing node
    if let Some(key) = state.state_key()
        && let Some(&existing) = tree.table.get(&key)
    {
        let val = match tree.nodes[existing].status {
            Status::Terminal(reward) => reward,
            _ => node_value(&tree.nodes[existing]),
        };
        return (existing, Some(val));
    }

    let status = state.status();
    let id = tree.nodes.len();

    if let Status::Terminal(reward) = status {
        tree.nodes.push(Node {
            status,
            is_chance: false,
            edges: Vec::new(),
        });
        return (id, Some(reward));
    }

    bufs.chances.clear();
    state.chance_outcomes(&mut bufs.chances);
    let is_chance = !bufs.chances.is_empty();

    let (edges, value) = if is_chance {
        let edges = bufs
            .chances
            .iter()
            .map(|&(outcome, prob)| Edge {
                action: outcome,
                child: None,
                prior: prob,
                visits: 0,
                total_value: 0.0,
            })
            .collect();
        (edges, None)
    } else {
        let nn = evaluator.evaluate(state);
        bufs.actions.clear();
        state.legal_actions(&mut bufs.actions);
        let priors = softmax_legal(&nn.policy_logits, &bufs.actions);
        let edges = bufs
            .actions
            .iter()
            .zip(priors)
            .map(|(&a, p)| Edge {
                action: a,
                child: None,
                prior: p,
                visits: 0,
                total_value: 0.0,
            })
            .collect();
        (edges, Some(nn.value))
    };

    tree.nodes.push(Node {
        status,
        is_chance,
        edges,
    });

    if let Some(key) = state.state_key() {
        tree.table.insert(key, id);
    }

    (id, value)
}

// ── Selection ─────────────────────────────────────────────────────────

fn select_edge(node: &Node, config: &Config) -> usize {
    let Status::Ongoing(player) = node.status else {
        unreachable!();
    };
    let sign = player.sign();
    let total_visits: u32 = node.edges.iter().map(|e| e.visits).sum();
    let sqrt_total = (total_visits as f32).sqrt();

    // FPU: unvisited edges use parent mean value minus a reduction
    let parent_q = if total_visits > 0 {
        node.edges.iter().map(|e| e.total_value).sum::<f32>() / total_visits as f32
    } else {
        0.0
    };
    let fpu_q = parent_q - config.fpu_reduction * sign;

    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    for (i, edge) in node.edges.iter().enumerate() {
        let q = if edge.visits > 0 {
            edge.total_value / edge.visits as f32
        } else {
            fpu_q
        };
        let u = config.cpuct * edge.prior * sqrt_total / (1.0 + edge.visits as f32);
        let score = sign * q + u;

        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    best_idx
}

fn sample_chance_edge(node: &Node, rng: &mut impl Rng) -> usize {
    let total: f32 = node.edges.iter().map(|e| e.prior).sum();
    let mut r = rng.random_range(0.0..total);
    for (i, edge) in node.edges.iter().enumerate() {
        r -= edge.prior;
        if r <= 0.0 {
            return i;
        }
    }
    node.edges.len() - 1
}

// ── Helpers ───────────────────────────────────────────────────────────

fn node_value(node: &Node) -> f32 {
    let total_visits: u32 = node.edges.iter().map(|e| e.visits).sum();
    if total_visits == 0 {
        return 0.0;
    }
    node.edges.iter().map(|e| e.total_value).sum::<f32>() / total_visits as f32
}

fn add_dirichlet_noise(edges: &mut [Edge], alpha: f32, epsilon: f32, rng: &mut impl Rng) {
    if edges.is_empty() || epsilon == 0.0 {
        return;
    }
    let gamma = Gamma::<f32>::new(alpha, 1.0).unwrap();
    let noise: Vec<f32> = edges.iter().map(|_| gamma.sample(rng)).collect();
    let sum: f32 = noise.iter().sum();
    if sum == 0.0 {
        return;
    }
    for (edge, &n) in edges.iter_mut().zip(&noise) {
        edge.prior = (1.0 - epsilon) * edge.prior + epsilon * (n / sum);
    }
}

fn softmax_legal(logits: &[f32], legal_actions: &[usize]) -> Vec<f32> {
    let max = legal_actions
        .iter()
        .map(|&a| logits[a])
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = legal_actions
        .iter()
        .map(|&a| (logits[a] - max).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::RolloutEvaluator;
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
    fn mcts_finds_winning_action() {
        let game = TrivialGame::new();
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
        let config = Config {
            num_simulations: 500,
            ..Default::default()
        };
        let mut rng = rand::rng();
        let result = search(&game, &evaluator, &config, &mut rng);
        assert!(
            result.policy[0] > result.policy[1],
            "MCTS should find that action 0 wins: policy = {:?}",
            result.policy
        );
    }
}
