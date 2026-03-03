use std::collections::HashMap;

use crate::eval::Evaluator;
use crate::game::{Game, Status};
use crate::graph::{DiGraph, NodeId};
use crate::player::Player;

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

struct Edge {
    action: usize,
    child: Option<NodeId>,
    prior: f32,
    visits: u32,
}

struct NodeData {
    status: Status,
    is_chance: bool,
    total_visits: u32,
    utility: f32,
    q: f32,
}

#[derive(Default)]
struct Tree {
    graph: DiGraph<NodeData, Edge>,
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
pub fn search<G: Game, E: Evaluator<G> + ?Sized>(
    root_state: &G,
    evaluator: &E,
    config: &Config,
    rng: &mut fastrand::Rng,
) -> SearchResult {
    if let Status::Terminal(reward) = root_state.status() {
        return SearchResult {
            policy: vec![0.0; G::NUM_ACTIONS],
            value: reward,
        };
    }

    let mut tree = Tree::default();
    let mut bufs = Bufs::default();

    let (root, _) = expand(&mut tree, root_state, evaluator, rng, &mut bufs);

    // Dirichlet noise on root priors (skip for chance roots)
    if !tree.graph[root].is_chance {
        add_dirichlet_noise(
            tree.graph.edges_mut(root),
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

    // Extract policy from root edge visit counts
    let edges = tree.graph.edges(root);
    let total_edge_visits: u32 = edges.iter().map(|e| e.visits).sum();
    let mut policy = vec![0.0f32; G::NUM_ACTIONS];
    if total_edge_visits > 0 {
        for edge in edges {
            policy[edge.action] = edge.visits as f32 / total_edge_visits as f32;
        }
    }

    SearchResult {
        policy,
        value: tree.graph[root].q,
    }
}

// ── Tree operations ───────────────────────────────────────────────────

fn simulate<G: Game, E: Evaluator<G> + ?Sized>(
    tree: &mut Tree,
    root: NodeId,
    root_state: &G,
    evaluator: &E,
    config: &Config,
    rng: &mut fastrand::Rng,
    bufs: &mut Bufs,
) {
    bufs.path.clear();
    let mut current = root;
    let mut state = root_state.clone();

    loop {
        let node = &tree.graph[current];
        let edges = tree.graph.edges(current);

        let edge_idx = match node.status {
            Status::Terminal(_) => break,
            Status::Ongoing(_) if node.is_chance => sample_chance_edge(edges, rng),
            Status::Ongoing(player) => puct_select(node, edges, &tree.graph, player, config),
        };

        bufs.path.push((current, edge_idx));

        let edge = &edges[edge_idx];
        let action = edge.action;
        let has_child = edge.child;

        state.apply_action(action);

        match has_child {
            Some(child) => current = child,
            None => {
                let (child_id, should_continue) = expand(tree, &state, evaluator, rng, bufs);
                tree.graph.edges_mut(current)[edge_idx].child = Some(child_id);
                if !should_continue {
                    break;
                }
                // Chance node — no eval, keep descending
                current = child_id;
            }
        }
    }

    // Backprop — idempotent Q recomputation
    for &(nid, eidx) in bufs.path.iter().rev() {
        tree.graph.edges_mut(nid)[eidx].visits += 1;

        let (sum_edge_visits, weighted_child_q) = {
            let edges = tree.graph.edges(nid);
            let sum = edges.iter().map(|e| e.visits).sum::<u32>();
            let mut wq = 0.0f32;
            for edge in edges {
                if let Some(child_id) = edge.child
                    && edge.visits > 0
                {
                    wq += edge.visits as f32 * tree.graph[child_id].q;
                }
            }
            (sum, wq)
        };

        let node = &mut tree.graph[nid];
        if node.is_chance {
            node.total_visits = sum_edge_visits;
            node.q = if sum_edge_visits > 0 {
                weighted_child_q / sum_edge_visits as f32
            } else {
                0.0
            };
        } else {
            node.total_visits = 1 + sum_edge_visits;
            node.q = (node.utility + weighted_child_q) / node.total_visits as f32;
        }
    }
}

fn expand<G: Game, E: Evaluator<G> + ?Sized>(
    tree: &mut Tree,
    state: &G,
    evaluator: &E,
    rng: &mut fastrand::Rng,
    bufs: &mut Bufs,
) -> (NodeId, bool) {
    // Transposition hit — reuse existing node
    if let Some(key) = state.state_key()
        && let Some(&existing) = tree.table.get(&key)
    {
        return (existing, false);
    }

    let status = state.status();

    if let Status::Terminal(reward) = status {
        let id = tree.graph.add_node(
            NodeData {
                status,
                is_chance: false,
                total_visits: 1,
                utility: reward,
                q: reward,
            },
            std::iter::empty(),
        );
        return (id, false);
    }

    state.chance_outcomes(&mut bufs.chances);
    let is_chance = !bufs.chances.is_empty();

    let (edges_iter, utility): (Box<dyn Iterator<Item = Edge>>, _) = if is_chance {
        let iter = bufs.chances.drain(..).map(|(action, prior)| Edge {
            action,
            child: None,
            prior,
            visits: 0,
        });
        (Box::new(iter), 0.0)
    } else {
        let nn = evaluator.evaluate(state, rng);
        state.legal_actions(&mut bufs.actions);
        let priors = softmax_legal(&nn.policy_logits, &bufs.actions);
        let iter = bufs
            .actions
            .drain(..)
            .zip(priors)
            .map(|(action, prior)| Edge {
                action,
                child: None,
                prior,
                visits: 0,
            });
        (Box::new(iter), nn.value)
    };

    let (total_visits, q) = if is_chance { (0, 0.0) } else { (1, utility) };

    let id = tree.graph.add_node(
        NodeData {
            status,
            is_chance,
            total_visits,
            utility,
            q,
        },
        edges_iter,
    );

    if let Some(key) = state.state_key() {
        tree.table.insert(key, id);
    }

    (id, is_chance)
}

// ── Selection ─────────────────────────────────────────────────────────

fn puct_select(
    node: &NodeData,
    edges: &[Edge],
    graph: &DiGraph<NodeData, Edge>,
    player: Player,
    config: &Config,
) -> usize {
    let sign = player.sign();
    let sqrt_total = (node.total_visits as f32).sqrt();

    let parent_q = sign * node.q;
    let fpu = parent_q - config.fpu_reduction;

    edges
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let q = if e.visits > 0 {
                sign * graph[e.child.unwrap()].q
            } else {
                fpu
            };
            let u = config.cpuct * e.prior * sqrt_total / (1.0 + e.visits as f32);
            (i, q + u)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .0
}

fn sample_chance_edge(edges: &[Edge], rng: &mut fastrand::Rng) -> usize {
    let total: f32 = edges.iter().map(|e| e.prior).sum();
    let mut r = rng.f32() * total;
    for (i, edge) in edges.iter().enumerate() {
        r -= edge.prior;
        if r <= 0.0 {
            return i;
        }
    }
    edges.len() - 1
}

// ── Helpers ───────────────────────────────────────────────────────────

fn add_dirichlet_noise(edges: &mut [Edge], alpha: f32, epsilon: f32, rng: &mut fastrand::Rng) {
    if edges.is_empty() || epsilon == 0.0 {
        return;
    }
    let noise = crate::dirichlet::sample(alpha, edges.len(), rng);
    for (edge, &n) in edges.iter_mut().zip(&noise) {
        edge.prior = (1.0 - epsilon) * edge.prior + epsilon * n;
    }
}

fn softmax_legal(logits: &[f32], legal_actions: &[usize]) -> Vec<f32> {
    let max = legal_actions
        .iter()
        .map(|&a| logits[a])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut priors: Vec<f32> = legal_actions
        .iter()
        .map(|&a| (logits[a] - max).exp())
        .collect();
    let sum: f32 = priors.iter().sum();
    priors.iter_mut().for_each(|p| *p /= sum);
    priors
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
        let mut rng = fastrand::Rng::new();
        let result = search(&game, &evaluator, &config, &mut rng);
        assert!(
            result.policy[0] > result.policy[1],
            "MCTS should find that action 0 wins: policy = {:?}",
            result.policy
        );
    }
}
