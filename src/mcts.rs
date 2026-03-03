use std::collections::HashMap;

use crate::eval::NnOutput;
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

/// One step of the MCTS state machine.
pub enum Step<G: Game> {
    /// Search needs an NN evaluation for this state.
    NeedsEval(G),
    /// Search complete.
    Done(SearchResult),
}

/// State machine for incremental MCTS search.
pub struct Search<G: Game> {
    tree: Tree,
    root: Option<NodeId>,
    root_state: G,
    bufs: Bufs,
    sims_remaining: u32,
    phase: Phase,
}

// ── Internal types ────────────────────────────────────────────────────

/// An outgoing edge from a tree node.
///
/// Invariant: `child.is_some()` iff `visits > 0`. An edge acquires both its
/// child node and its first visit during expansion; neither is set without
/// the other.
struct Edge {
    action: usize,
    child: Option<NodeId>,
    /// Policy prior used during search. Includes Dirichlet noise at the root.
    prior: f32,
    /// Original NN policy prior, without noise. Kept so that `resume` can
    /// re-apply fresh noise without compounding on previously noised values.
    raw_prior: f32,
    visits: u32,
}

impl Edge {
    fn new((action, prior): (usize, f32)) -> Self {
        Self {
            action,
            child: None,
            prior,
            raw_prior: prior,
            visits: 0,
        }
    }
}

/// MCTS-internal classification of a tree node.
enum NodeKind {
    Terminal,
    Decision(Player),
    Chance,
}

struct NodeData {
    kind: NodeKind,
    total_visits: u32,
    utility: f32,
    q: f32,
}

impl NodeData {
    /// Create a new node. `value` is the initial evaluation:
    /// - Terminal: the game reward.
    /// - Decision: the NN value estimate.
    /// - Chance: unused (0.0); Q is derived from children during backprop.
    ///
    /// Both `utility` (the node's own value, used in MCGS backprop) and `q`
    /// (the mixed value) start equal to `value`. Chance nodes start with 0
    /// visits since their visit count comes purely from edge sums; terminal
    /// and decision nodes start with 1 to account for their own evaluation.
    fn new(kind: NodeKind, value: f32) -> Self {
        let total_visits = match kind {
            NodeKind::Chance => 0,
            _ => 1,
        };
        Self {
            kind,
            total_visits,
            utility: value,
            q: value,
        }
    }
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

enum Phase {
    ExpandingRoot {
        player: Player,
    },
    Simulating {
        root: NodeId,
        parent: NodeId,
        edge_idx: usize,
        player: Player,
        state_key: Option<u64>,
    },
}

/// Result of `try_expand`: what was found at the leaf.
enum ExpandResult {
    /// Terminal or transposition — link and stop descending.
    Leaf(NodeId),
    /// Chance node — link and keep descending.
    Chance(NodeId),
    /// Decision node — needs NN evaluation. Carries the acting player.
    NeedsEval(Player),
}

enum SimResult<G: Game> {
    Complete,
    NeedsEval {
        state: G,
        parent: NodeId,
        edge_idx: usize,
        player: Player,
        state_key: Option<u64>,
    },
}

// ── Public API ────────────────────────────────────────────────────────

impl<G: Game> Search<G> {
    /// Begin a fresh search. Returns `(Self, Step)` so the caller can
    /// immediately act on the first step without a separate call — the root
    /// always either needs an eval or is already terminal.
    pub fn new(root_state: &G, config: &Config, rng: &mut fastrand::Rng) -> (Self, Step<G>) {
        let mut search = Self {
            tree: Tree::default(),
            root: None,
            root_state: root_state.clone(),
            bufs: Bufs::default(),
            sims_remaining: config.num_simulations,
            phase: Phase::ExpandingRoot {
                player: Player::One,
            }, // overwritten below
        };

        // Terminal root — immediate result
        if let Status::Terminal(reward) = root_state.status() {
            let step = Step::Done(SearchResult {
                policy: vec![0.0; G::NUM_ACTIONS],
                value: reward,
            });
            return (search, step);
        }

        // Try to expand root
        match try_expand(&mut search.tree, root_state, &mut search.bufs) {
            ExpandResult::Leaf(_) => unreachable!("empty non-terminal tree"),
            ExpandResult::Chance(id) => {
                search.root = Some(id);
                let step = search.run_simulations(id, config, rng);
                (search, step)
            }
            ExpandResult::NeedsEval(player) => {
                search.phase = Phase::ExpandingRoot { player };
                let step = Step::NeedsEval(root_state.clone());
                (search, step)
            }
        }
    }

    /// Supply the NN evaluation requested by a prior `NeedsEval` step.
    /// Must be called exactly once per `NeedsEval` — the search cannot
    /// progress without it because node expansion requires policy priors.
    pub fn supply(&mut self, eval: NnOutput, config: &Config, rng: &mut fastrand::Rng) -> Step<G> {
        let root = match self.phase {
            Phase::ExpandingRoot { player } => {
                let state_key = self.root_state.state_key();
                let root =
                    complete_expand(&mut self.tree, &eval, &mut self.bufs, player, state_key);
                self.root = Some(root);
                add_dirichlet_noise(
                    self.tree.graph.edges_mut(root),
                    config.dirichlet_alpha,
                    config.dirichlet_epsilon,
                    rng,
                );
                root
            }
            Phase::Simulating {
                root,
                parent,
                edge_idx,
                player,
                state_key,
            } => {
                let child =
                    complete_expand(&mut self.tree, &eval, &mut self.bufs, player, state_key);
                self.tree.graph.edges_mut(parent)[edge_idx].child = Some(child);
                backprop(&mut self.tree, &self.bufs.path);
                self.sims_remaining -= 1;
                root
            }
        };
        self.run_simulations(root, config, rng)
    }

    /// Walk the tree root to the child of `action`.
    /// Call once per action applied to the game state between searches.
    /// No-op if the child doesn't exist in the tree.
    pub fn advance(&mut self, action: usize) {
        let Some(root) = self.root else { return };
        let edges = self.tree.graph.edges(root);
        let child = edges
            .iter()
            .find(|e| e.action == action)
            .and_then(|e| e.child);
        self.root = child;
    }

    /// Start a new search, reusing the existing subtree if available.
    /// Compacts the tree via `retain_subtree`, clears the transposition table,
    /// resets `sims_remaining`, and applies Dirichlet noise to the new root.
    /// Falls back to `Search::new` if the root was lost during `advance`.
    pub fn resume(&mut self, root_state: &G, config: &Config, rng: &mut fastrand::Rng) -> Step<G> {
        let Some(old_root) = self.root else {
            let (new_search, step) = Search::new(root_state, config, rng);
            *self = new_search;
            return step;
        };

        // Compact the graph
        let new_root = self.tree.graph.retain_subtree(
            old_root,
            |e| e.child,
            |e, new_child| e.child = new_child,
        );
        self.root = Some(new_root);
        self.root_state = root_state.clone();
        self.tree.table.clear();
        self.sims_remaining = config.num_simulations;

        // Terminal root — immediate result
        if let Status::Terminal(reward) = root_state.status() {
            return Step::Done(SearchResult {
                policy: vec![0.0; G::NUM_ACTIONS],
                value: reward,
            });
        }

        // Add Dirichlet noise to the (reused) root
        add_dirichlet_noise(
            self.tree.graph.edges_mut(new_root),
            config.dirichlet_alpha,
            config.dirichlet_epsilon,
            rng,
        );

        self.run_simulations(new_root, config, rng)
    }

    fn run_simulations(
        &mut self,
        root: NodeId,
        config: &Config,
        rng: &mut fastrand::Rng,
    ) -> Step<G> {
        while self.sims_remaining > 0 {
            match simulate_one(
                &mut self.tree,
                root,
                &self.root_state,
                config,
                rng,
                &mut self.bufs,
            ) {
                SimResult::Complete => {
                    self.sims_remaining -= 1;
                }
                SimResult::NeedsEval {
                    state,
                    parent,
                    edge_idx,
                    player,
                    state_key,
                } => {
                    self.phase = Phase::Simulating {
                        root,
                        parent,
                        edge_idx,
                        player,
                        state_key,
                    };
                    return Step::NeedsEval(state);
                }
            }
        }
        Step::Done(extract_result::<G>(&self.tree, root))
    }
}

// ── Tree operations ───────────────────────────────────────────────────

/// Run one MCTS simulation: select down from the root, expand a leaf, and
/// backpropagate. Returns `NeedsEval` if the leaf is a decision node (the
/// caller must supply an NN eval before this simulation can finish).
fn simulate_one<G: Game>(
    tree: &mut Tree,
    root: NodeId,
    root_state: &G,
    config: &Config,
    rng: &mut fastrand::Rng,
    bufs: &mut Bufs,
) -> SimResult<G> {
    bufs.path.clear();
    let mut current = root;
    let mut state = root_state.clone();

    loop {
        let node = &tree.graph[current];
        let edges = tree.graph.edges(current);

        let edge_idx = match node.kind {
            NodeKind::Terminal => break,
            NodeKind::Chance => sample_chance_edge(edges, rng),
            NodeKind::Decision(player) => puct_select(node, edges, &tree.graph, player, config),
        };

        bufs.path.push((current, edge_idx));
        let action = edges[edge_idx].action;
        let child_opt = edges[edge_idx].child;
        state.apply_action(action);

        // Existing child — descend
        if let Some(child) = child_opt {
            current = child;
            continue;
        }

        // Expand leaf
        match try_expand(tree, &state, bufs) {
            ExpandResult::NeedsEval(player) => {
                let state_key = state.state_key();
                return SimResult::NeedsEval {
                    state,
                    parent: current,
                    edge_idx,
                    player,
                    state_key,
                };
            }
            ExpandResult::Chance(id) => {
                tree.graph.edges_mut(current)[edge_idx].child = Some(id);
                current = id;
                // keep descending through chance
            }
            ExpandResult::Leaf(id) => {
                tree.graph.edges_mut(current)[edge_idx].child = Some(id);
                break;
            }
        }
    }

    backprop(tree, &bufs.path);
    SimResult::Complete
}

/// Attempt to expand a leaf state into the tree. Handles transpositions,
/// terminals, and chance nodes immediately; decision nodes require an NN
/// eval so they return `NeedsEval` with `bufs.actions` pre-populated for
/// the subsequent `complete_expand` call.
fn try_expand<G: Game>(tree: &mut Tree, state: &G, bufs: &mut Bufs) -> ExpandResult {
    let state_key = state.state_key();

    // Transposition hit — reuse existing node
    if let Some(&existing) = state_key.and_then(|k| tree.table.get(&k)) {
        return ExpandResult::Leaf(existing);
    }

    match state.status() {
        Status::Terminal(reward) => {
            let id = insert_node(
                tree,
                state_key,
                NodeKind::Terminal,
                reward,
                std::iter::empty(),
            );
            ExpandResult::Leaf(id)
        }
        Status::Ongoing(player) => {
            state.chance_outcomes(&mut bufs.chances);
            if !bufs.chances.is_empty() {
                let edges = bufs.chances.drain(..).map(Edge::new);
                let id = insert_node(tree, state_key, NodeKind::Chance, 0.0, edges);
                return ExpandResult::Chance(id);
            }

            bufs.actions.clear();
            state.legal_actions(&mut bufs.actions);
            ExpandResult::NeedsEval(player)
        }
    }
}

/// Finish expanding a decision node after receiving its NN evaluation.
/// Consumes `bufs.actions` (populated by the prior `try_expand` call) and
/// the eval's policy logits to create the node with softmax priors.
fn complete_expand(
    tree: &mut Tree,
    eval: &NnOutput,
    bufs: &mut Bufs,
    player: Player,
    state_key: Option<u64>,
) -> NodeId {
    let priors = crate::utils::softmax_masked(&eval.policy_logits, &bufs.actions);
    let edges = bufs.actions.drain(..).zip(priors).map(Edge::new);
    insert_node(
        tree,
        state_key,
        NodeKind::Decision(player),
        eval.value,
        edges,
    )
}

/// Add a node to the tree graph and register it in the transposition table.
fn insert_node(
    tree: &mut Tree,
    state_key: Option<u64>,
    kind: NodeKind,
    value: f32,
    edges: impl Iterator<Item = Edge>,
) -> NodeId {
    let id = tree.graph.add_node(NodeData::new(kind, value), edges);
    if let Some(key) = state_key {
        tree.table.insert(key, id);
    }
    id
}

/// KataGo-style MCGS backpropagation. Walks the path from leaf to root,
/// recomputing each node's Q as a weighted average of its own utility and
/// its children's Q values. This is parent-path-only — transposition nodes
/// reached from other parents are not updated here (full DAG-aware backprop
/// is too expensive).
fn backprop(tree: &mut Tree, path: &[(NodeId, usize)]) {
    for &(nid, eidx) in path.iter().rev() {
        tree.graph.edges_mut(nid)[eidx].visits += 1;

        let (sum_edge_visits, weighted_child_q) = {
            let edges = tree.graph.edges(nid);
            let sum = edges.iter().map(|e| e.visits).sum::<u32>();
            let mut wq = 0.0f32;
            for edge in edges {
                if let Some(child_id) = edge.child {
                    wq += edge.visits as f32 * tree.graph[child_id].q;
                }
            }
            (sum, wq)
        };

        let node = &mut tree.graph[nid];
        match node.kind {
            NodeKind::Chance => {
                node.total_visits = sum_edge_visits;
                node.q = if sum_edge_visits > 0 {
                    weighted_child_q / sum_edge_visits as f32
                } else {
                    0.0
                };
            }
            _ => {
                node.total_visits = 1 + sum_edge_visits;
                node.q = (node.utility + weighted_child_q) / node.total_visits as f32;
            }
        }
    }
}

/// Build the search result from the root's edge visit counts. The policy
/// is proportional to visits (not softmax of Q) because visit counts
/// already reflect the search's confidence after exploration.
fn extract_result<G: Game>(tree: &Tree, root: NodeId) -> SearchResult {
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

// ── Selection ─────────────────────────────────────────────────────────

/// Select the edge with highest PUCT score: Q + c * prior * sqrt(N_parent) / (1 + N_edge).
/// Unvisited edges use first-play urgency (parent Q minus a reduction) so
/// they're explored before low-Q visited edges but after high-Q ones.
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
            debug_assert_eq!(e.child.is_some(), e.visits > 0);
            let q = e.child.map_or(fpu, |c| sign * graph[c].q);
            let u = config.cpuct * e.prior * sqrt_total / (1.0 + e.visits as f32);
            (i, q + u)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .0
}

/// Sample a chance edge proportional to its prior (the game-provided
/// probability). No exploration bonus — chance outcomes are sampled, not chosen.
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

/// Mix Dirichlet noise into the root's priors to encourage exploration of
/// moves the NN might undervalue. Blends from `raw_prior` (not `prior`)
/// so repeated calls (e.g. across `resume`) don't compound noise.
fn add_dirichlet_noise(edges: &mut [Edge], alpha: f32, epsilon: f32, rng: &mut fastrand::Rng) {
    if edges.is_empty() || epsilon == 0.0 {
        return;
    }
    let noise = crate::dirichlet::sample(alpha, edges.len(), rng);
    for (edge, &n) in edges.iter_mut().zip(&noise) {
        edge.prior = (1.0 - epsilon) * edge.raw_prior + epsilon * n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Evaluator, RolloutEvaluator};
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

        let (mut sm, mut step) = Search::new(&game, &config, &mut rng);
        let result = loop {
            step = match step {
                Step::NeedsEval(s) => {
                    let output = evaluator.evaluate(&s, &mut rng);
                    sm.supply(output, &config, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

        assert!(
            result.policy[0] > result.policy[1],
            "MCTS should find that action 0 wins: policy = {:?}",
            result.policy
        );
    }

    #[test]
    fn state_machine_api_works() {
        let game = TrivialGame::new();
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
        let config = Config {
            num_simulations: 500,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let (mut sm, mut step) = Search::new(&game, &config, &mut rng);
        let result = loop {
            step = match step {
                Step::NeedsEval(s) => {
                    let output = evaluator.evaluate(&s, &mut rng);
                    sm.supply(output, &config, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

        assert!(
            result.policy[0] > result.policy[1],
            "State machine API should find that action 0 wins: policy = {:?}",
            result.policy
        );
    }

    /// Two-step game: P1 picks action 0 or 1, then picks 0 or 1 again.
    /// Win (+1) only if both actions are 0.
    #[derive(Clone)]
    struct TwoStepGame {
        step: u8,
        chose_zero_first: bool,
        reward: Option<f32>,
    }

    impl TwoStepGame {
        fn new() -> Self {
            Self {
                step: 0,
                chose_zero_first: false,
                reward: None,
            }
        }
    }

    impl Game for TwoStepGame {
        const NUM_ACTIONS: usize = 2;

        fn status(&self) -> Status {
            match self.reward {
                Some(r) => Status::Terminal(r),
                None => Status::Ongoing(Player::One),
            }
        }
        fn legal_actions(&self, buf: &mut Vec<usize>) {
            if self.reward.is_none() {
                buf.push(0);
                buf.push(1);
            }
        }
        fn apply_action(&mut self, action: usize) {
            match self.step {
                0 => {
                    self.chose_zero_first = action == 0;
                    self.step = 1;
                }
                1 => {
                    self.reward = Some(if self.chose_zero_first && action == 0 {
                        1.0
                    } else {
                        -1.0
                    });
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn tree_reuse_works() {
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
        let config = Config {
            num_simulations: 200,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        // First search from the initial state
        let mut game = TwoStepGame::new();
        let (mut search, mut step) = Search::new(&game, &config, &mut rng);
        let result = loop {
            step = match step {
                Step::NeedsEval(s) => {
                    let output = evaluator.evaluate(&s, &mut rng);
                    search.supply(output, &config, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

        // Apply the chosen action
        let action = result
            .policy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        game.apply_action(action);
        search.advance(action);

        // Resume search from the new state
        let node_count_before = search.tree.graph.node_count();
        step = search.resume(&game, &config, &mut rng);
        let node_count_after_compact = search.tree.graph.node_count();
        assert!(
            node_count_after_compact < node_count_before,
            "retain_subtree should compact: {node_count_after_compact} >= {node_count_before}"
        );

        let result2 = loop {
            step = match step {
                Step::NeedsEval(s) => {
                    let output = evaluator.evaluate(&s, &mut rng);
                    search.supply(output, &config, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

        // Second search should produce a valid policy
        let total: f32 = result2.policy.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "policy should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn state_machine_terminal_root() {
        let game = TrivialGame {
            done: true,
            chose_win: true,
        };
        let config = Config::default();
        let mut rng = fastrand::Rng::new();

        let (_sm, step) = Search::new(&game, &config, &mut rng);
        match step {
            Step::Done(result) => {
                assert_eq!(result.value, 1.0);
                assert!(result.policy.iter().all(|&p| p == 0.0));
            }
            Step::NeedsEval(_) => panic!("terminal root should not need eval"),
        }
    }
}
