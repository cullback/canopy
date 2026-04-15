mod tree;

use crate::eval::{Evaluation, Wdl};
use crate::game::{Game, Status};

use tree::{Bufs, ExpandResult};
pub use tree::{Edge, NodeId, NodeKind, Tree};

// ── Public types ──────────────────────────────────────────────────────

/// PUCT MCTS configuration.
#[derive(Clone)]
pub struct Config {
    /// Simulation budget per search. Set to 0 to skip search entirely and
    /// select an action from the evaluator's policy alone (useful for
    /// heuristic evaluators that encode a fixed strategy).
    pub num_simulations: u32,
    /// σ scaling parameter (controls Q influence on improved policy).
    pub c_visit: f32,
    /// σ scaling parameter.
    pub c_scale: f32,
    /// PUCT exploration constant.
    pub c_puct: f32,
    /// Dirichlet noise alpha (concentration parameter).
    pub dirichlet_alpha: f32,
    /// Dirichlet noise weight mixed into root priors.
    pub dirichlet_epsilon: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            c_visit: 50.0,
            c_scale: 1.0,
            c_puct: 2.5,
            dirichlet_alpha: 0.05,
            dirichlet_epsilon: 0.25,
        }
    }
}

/// Result of an MCTS search.
pub struct SearchResult {
    /// Improved policy over `[0, NUM_ACTIONS)` (training target).
    pub policy: Vec<f32>,
    /// Root WDL (P1 perspective).
    pub wdl: Wdl,
    /// The action selected by search to play.
    pub selected_action: usize,
    /// Raw network value (P1 perspective) before search corrections.
    pub network_value: f32,
    /// (action, Q) pairs for visited root children.
    pub children_q: Vec<(usize, f32)>,
    /// Action with highest raw prior (network policy argmax).
    pub prior_top1_action: usize,
    /// Principal variation depth: edges along the most-visited path from
    /// root to the deepest expanded node.
    pub pv_depth: u32,
    /// Maximum simulation depth across all sims.
    pub max_depth: u32,
}

/// Leaf identifier returned by [`Search::select`].
///
/// Opaque handle — pass back to [`Search::backup`] or
/// [`Search::backup_terminal`] to propagate the evaluation.
pub struct LeafId {
    inner: LeafInner,
}

enum LeafInner {
    /// Root needs expansion: tree was empty, now the root decision node
    /// needs an evaluation before simulations can start.
    RootExpansion { sign: f32, actions: Vec<usize> },
    /// Normal simulation leaf: descended from root, hit an unexpanded
    /// decision node.
    Simulation {
        sign: f32,
        actions: Vec<usize>,
        path: Vec<(NodeId, usize)>,
        state_key: Option<u64>,
    },
    /// Terminal leaf: already handled, backup is a no-op (value was
    /// backpropagated synchronously during select).
    Terminal,
}

/// One step of the MCTS select phase.
pub enum Select<G: Game> {
    /// Leaf needs evaluation — evaluate the state and pass to
    /// [`Search::backup`].
    Eval(LeafId, G),
    /// Terminal leaf — pass to [`Search::backup_terminal`] or ignore
    /// (value is already propagated).
    Terminal(LeafId, Wdl),
    /// Search complete (simulation budget exhausted).
    Done,
}

/// PUCT MCTS, driven as a state machine.
///
/// The search tree persists across actions so that subtrees explored in earlier
/// searches can be reused (the tree is compacted, not rebuilt, on each new
/// search).  The public methods form a simple protocol:
///
/// 1. [`new`](Self::new) — construct with a root game state and config.
/// 2. [`select`](Self::select) — descend from root, returning a leaf.
/// 3. [`backup`](Self::backup) — propagate an evaluation.
/// 4. [`backup_terminal`](Self::backup_terminal) — propagate a terminal value.
/// 5. [`result`](Self::result) — extract the search result when done.
/// 6. [`apply_action`](Self::apply_action) — mirror actions as they happen.
pub struct Search<G: Game> {
    tree: Tree,
    /// `None` initially or after `apply_action` walks through an unexpanded
    /// child.  The next `select` will discard the old tree and start fresh.
    root: Option<NodeId>,
    root_state: G,
    bufs: Bufs,
    config: Config,
    /// Simulation count for the current search.
    sims_done: u32,
    /// Q bounds for normalization, widened after each sim.
    q_bounds: (f32, f32),
    /// Dirichlet noise for root priors.
    root_noise: Vec<f32>,
    /// Raw network value for the root (P1 perspective), captured on first
    /// expansion and reused in the `SearchResult`.
    root_network_value: f32,
    /// Maximum simulation depth across all sims in the current search.
    depth_max: u32,
    /// Whether the tree needs compaction before starting the next search.
    needs_compact: bool,
}

// ── Public API ────────────────────────────────────────────────────────

impl<G: Game> Search<G> {
    /// Create a new search with the given root game state and config.
    ///
    /// The tree starts empty; the first call to [`select`](Self::select)
    /// will expand from scratch.
    pub fn new(root_state: G, config: Config) -> Self {
        Self {
            tree: Tree::default(),
            root: None,
            root_state,
            bufs: Bufs::default(),
            config,
            sims_done: 0,
            q_bounds: (0.0, 0.0),
            root_noise: Vec::new(),
            root_network_value: 0.0,
            depth_max: 0,
            needs_compact: false,
        }
    }

    /// Reset for a new game, reusing internal allocations.
    pub fn reset(&mut self, root_state: G) {
        self.tree.clear();
        self.root = None;
        self.root_state = root_state;
        self.sims_done = 0;
        self.q_bounds = (0.0, 0.0);
        self.root_noise.clear();
        self.root_network_value = 0.0;
        self.depth_max = 0;
        self.needs_compact = false;
    }

    /// Read access to the internal game state.
    pub fn state(&self) -> &G {
        &self.root_state
    }

    /// Read access to the internal search tree.
    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Mutate the root state without clearing the tree.
    ///
    /// Use for minor corrections (e.g. robber position, turn flags) that
    /// don't invalidate the search tree's structure.
    pub fn update_state(&mut self, f: impl FnOnce(&mut G)) {
        f(&mut self.root_state);
    }

    /// Cancel any in-progress search, cleaning up virtual losses.
    ///
    /// Safe to call even when no search is active (no-op).
    pub fn cancel_search(&mut self) {
        self.sims_done = 0;
    }

    /// Update the simulation budget (takes effect on the next search).
    pub fn set_num_simulations(&mut self, n: u32) {
        self.config.num_simulations = n;
    }

    /// Total visits on the current root (sum of edge visit counts), or 0 if
    /// the root is not yet expanded.
    pub fn root_visits(&self) -> u32 {
        self.root
            .map(|r| self.tree.edges(r).iter().map(|e| e.visits).sum())
            .unwrap_or(0)
    }

    /// Read access to the MCTS config.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Apply an action to the internal game state and walk the tree pointer.
    ///
    /// If the tree has an expanded child for `action`, the root pointer follows
    /// it (O(1)).  Otherwise the root pointer becomes `None` and the next
    /// search will discard the old tree and start fresh.
    pub fn apply_action(&mut self, action: usize) {
        self.root_state.apply_action(action);
        if let Some(root) = self.root {
            self.root = self.tree.child_for_action(root, action);
        }
        self.needs_compact = true;
    }

    /// Walk the tree pointer through a sequence of actions without touching
    /// `root_state`.
    ///
    /// Use when the caller will set `root_state` separately (e.g. via
    /// `update_state`). At chance nodes, falls back to the most-visited
    /// child if the exact outcome wasn't explored or no action was
    /// recorded for it. Stops early once the pointer becomes `None`.
    pub fn walk_tree(&mut self, actions: &[usize]) -> usize {
        let mut walked = 0;
        for &action in actions {
            let Some(root) = self.root else { break };
            match self.tree.child_for_action(root, action) {
                Some(child) => {
                    self.root = Some(child);
                    walked += 1;
                }
                None => {
                    if matches!(*self.tree.kind(root), NodeKind::Chance) {
                        self.root = self.tree.best_chance_child(root);
                        walked += 1;
                    } else {
                        let edges = self.tree.edges(root);
                        let edge_actions: Vec<usize> = edges.iter().map(|e| e.action).collect();
                        eprintln!(
                            "walk_tree: no child for action {action} at depth {walked} \
                             (node has {} edges: {:?})",
                            edges.len(),
                            edge_actions,
                        );
                        break;
                    }
                }
            }
        }
        // Skip past any trailing chance node (e.g. BUY_DEV_CARD lands on
        // DevCardDraw but no outcome action was recorded).
        if let Some(root) = self.root
            && matches!(*self.tree.kind(root), NodeKind::Chance)
        {
            self.root = self.tree.best_chance_child(root);
        }
        self.needs_compact = true;
        walked
    }

    /// Select a leaf node by descending from root via PUCT.
    ///
    /// Returns [`Select::Eval`] with a leaf state needing evaluation,
    /// [`Select::Terminal`] with a terminal value, or [`Select::Done`]
    /// when the simulation budget is exhausted.
    pub fn select(&mut self, rng: &mut fastrand::Rng) -> Select<G> {
        // Budget exhausted — search is done.
        if self.sims_done >= self.config.num_simulations {
            return Select::Done;
        }

        // Ensure the root is expanded.
        if self.root.is_none() {
            return self.expand_root(rng);
        }

        // Compact tree from a previous apply_action / walk_tree.
        if self.needs_compact {
            self.begin_search(rng);
        }

        // Run one simulation from root.
        self.simulate(rng)
    }

    /// Propagate an evaluation for a previously returned leaf.
    pub fn backup(&mut self, leaf: LeafId, eval: Evaluation) {
        match leaf.inner {
            LeafInner::RootExpansion { sign, actions } => {
                let state_key = self.root_state.state_key();
                let root = self.tree.complete_expand(&eval, &actions, sign, state_key);
                self.root = Some(root);
                self.root_network_value = eval.wdl.q();
                // Dirichlet noise is generated lazily on the next select() call.
                self.root_noise.clear();
                self.bufs.reclaim_actions(actions);
            }
            LeafInner::Simulation {
                sign,
                actions,
                path,
                state_key,
            } => {
                self.tree.remove_virtual_loss(&path);
                let &(parent, edge_idx) = path.last().unwrap();
                if self.tree.edges(parent)[edge_idx].child.is_none() {
                    let child = match state_key.and_then(|k| self.tree.lookup(k)) {
                        Some(existing) => existing,
                        None => self.tree.complete_expand(&eval, &actions, sign, state_key),
                    };
                    self.tree.set_child(parent, edge_idx, child);
                }
                self.tree.recompute_q(&path);
                widen_q_bounds(&self.tree, &path, &mut self.q_bounds);
                self.bufs.reclaim_path(path);
                self.bufs.reclaim_actions(actions);
            }
            LeafInner::Terminal => {}
        }
    }

    /// Propagate a terminal value for a previously returned leaf.
    pub fn backup_terminal(&mut self, leaf: LeafId, _wdl: Wdl) {
        // Terminal values are already handled during select — the tree was
        // backpropagated synchronously. This is a no-op.
        match leaf.inner {
            LeafInner::Terminal => {}
            _ => {
                // Shouldn't happen, but handle gracefully.
            }
        }
    }

    /// Extract the search result when the simulation budget is exhausted.
    pub fn result(&self) -> SearchResult {
        let root = match self.root {
            Some(r) => r,
            None => {
                // No root expanded — terminal or degenerate state.
                return SearchResult {
                    policy: vec![0.0; G::NUM_ACTIONS],
                    wdl: Wdl::DRAW,
                    selected_action: 0,
                    network_value: 0.0,
                    children_q: vec![],
                    prior_top1_action: 0,
                    pv_depth: 0,
                    max_depth: 0,
                };
            }
        };

        let edges = self.tree.edges(root);
        let root_sign = match *self.tree.kind(root) {
            NodeKind::Decision(sign) => sign,
            _ => 1.0,
        };

        // Improved policy: softmax(logit + σ(completedQ)) over edges.
        let max_visits = self.tree.max_edge_visits(root);
        let vmix_val = v_mix(&self.tree, root);
        let mut improved_logits = Vec::with_capacity(edges.len());
        for edge in edges {
            let cq = completed_q(&self.tree, edge, vmix_val);
            let q_norm = normalize_q(cq, self.q_bounds.0, self.q_bounds.1, root_sign);
            let s = sigma(q_norm, max_visits, self.config.c_visit, self.config.c_scale);
            improved_logits.push(edge.logit + s);
        }
        softmax(&mut improved_logits);

        let mut policy = vec![0.0f32; G::NUM_ACTIONS];
        for (edge, &prob) in edges.iter().zip(&improved_logits) {
            policy[edge.action] = prob;
        }

        // Selected action: highest visit count.
        let selected_action = edges
            .iter()
            .max_by_key(|e| e.visits)
            .map(|e| e.action)
            .unwrap_or(0);

        let prior_top1_action = edges
            .iter()
            .max_by(|a, b| a.logit.total_cmp(&b.logit))
            .map(|e| e.action)
            .unwrap_or(selected_action);

        let children_q: Vec<(usize, f32)> = edges
            .iter()
            .filter_map(|e| e.child.map(|c| (e.action, self.tree.q(c))))
            .collect();

        let pv_depth = compute_pv_depth(&self.tree, root);

        SearchResult {
            policy,
            wdl: self.tree.wdl(root),
            selected_action,
            network_value: self.root_network_value,
            children_q,
            prior_top1_action,
            pv_depth,
            max_depth: self.depth_max,
        }
    }

    // ── Internal ─────────────────────────────────────────────────────

    /// Begin a new search: compact tree, reset counters, generate noise.
    fn begin_search(&mut self, rng: &mut fastrand::Rng) {
        self.needs_compact = false;
        self.sims_done = 0;
        self.q_bounds = (0.0, 0.0);
        self.depth_max = 0;

        if let Some(old_root) = self.root {
            let new_root = self.tree.compact(old_root);
            self.root = Some(new_root);
            self.root_network_value = self.tree.utility(new_root);

            // Regenerate Dirichlet noise for the (possibly new) root.
            self.root_noise = sample_dirichlet(
                self.config.dirichlet_alpha,
                self.tree.edges(new_root).len(),
                rng,
            );
        }
    }

    /// First-time root expansion (tree is empty).
    fn expand_root(&mut self, rng: &mut fastrand::Rng) -> Select<G> {
        self.needs_compact = false;
        self.sims_done = 0;
        self.q_bounds = (0.0, 0.0);
        self.depth_max = 0;

        // Terminal root — immediate result.
        if let Status::Terminal(reward) = self.root_state.status() {
            self.sims_done = self.config.num_simulations;
            return Select::Terminal(
                LeafId {
                    inner: LeafInner::Terminal,
                },
                Wdl::from_value(reward),
            );
        }

        // Clear nodes but preserve allocations.
        self.tree.clear();

        match self.tree.try_expand(&self.root_state, &mut self.bufs) {
            ExpandResult::Leaf(id) => {
                // Terminal or transposition hit. For terminal: value already stored.
                self.root = Some(id);
                self.root_network_value = self.tree.utility(id);
                self.root_noise =
                    sample_dirichlet(self.config.dirichlet_alpha, self.tree.edges(id).len(), rng);
                // Count as one sim and continue.
                self.sims_done += 1;
                self.simulate(rng)
            }
            ExpandResult::Chance(id) => {
                self.root = Some(id);
                self.root_network_value = self.tree.utility(id);
                // Chance root — proceed to simulation.
                self.simulate(rng)
            }
            ExpandResult::NeedsEval(sign) => {
                let state = self.root_state.clone();
                Select::Eval(
                    LeafId {
                        inner: LeafInner::RootExpansion {
                            sign,
                            actions: self.bufs.take_actions(),
                        },
                    },
                    state,
                )
            }
        }
    }

    /// Run one simulation from root, returning the leaf.
    fn simulate(&mut self, rng: &mut fastrand::Rng) -> Select<G> {
        let root = match self.root {
            Some(r) => r,
            None => {
                self.sims_done = self.config.num_simulations;
                return Select::Done;
            }
        };

        // Lazy Dirichlet noise generation (after root expansion or compaction).
        let num_edges = self.tree.edges(root).len();
        if self.root_noise.len() != num_edges && num_edges > 0 {
            self.root_noise = sample_dirichlet(self.config.dirichlet_alpha, num_edges, rng);
        }

        self.bufs.path.clear();
        let mut current = root;
        let mut state = self.root_state.clone();
        let is_imperfect = state.determinize(rng);

        loop {
            let edge_idx = match *self.tree.kind(current) {
                NodeKind::Terminal => break,
                NodeKind::Chance => {
                    if is_imperfect {
                        // SO-ISMCTS: resample from determinized state.
                        match state.sample_chance(rng) {
                            Some(outcome) => {
                                let edges = self.tree.edges(current);
                                match edges.iter().position(|e| e.action == outcome) {
                                    Some(idx) => idx,
                                    None => break,
                                }
                            }
                            None => break,
                        }
                    } else {
                        self.tree.sample_chance_edge(current, rng)
                    }
                }
                NodeKind::Decision(sign) => {
                    let edges = self.tree.edges(current);
                    if is_imperfect {
                        self.bufs.legal.clear();
                        state.legal_actions(&mut self.bufs.legal);
                        self.bufs.legal_edges.clear();
                        for (i, edge) in edges.iter().enumerate() {
                            if self.bufs.legal.contains(&edge.action) {
                                self.bufs.legal_edges.push(i);
                            }
                        }
                        if self.bufs.legal_edges.is_empty() {
                            break;
                        }
                    } else {
                        self.bufs.legal_edges.clear();
                        self.bufs.legal_edges.extend(0..edges.len());
                    }

                    let is_root = current == root;
                    let noise = if is_root && !self.root_noise.is_empty() {
                        Some(&self.root_noise[..])
                    } else {
                        None
                    };

                    if is_root {
                        puct_select(
                            &self.tree,
                            current,
                            sign,
                            &self.config,
                            &self.bufs.legal_edges,
                            noise,
                        )
                    } else {
                        interior_select(
                            &self.tree,
                            current,
                            sign,
                            &self.config,
                            self.q_bounds,
                            &mut self.bufs.scratch,
                            &self.bufs.legal_edges,
                        )
                    }
                }
            };

            let edges = self.tree.edges(current);
            self.bufs.path.push((current, edge_idx));
            let action = edges[edge_idx].action;
            let child_opt = edges[edge_idx].child;

            state.apply_action(action);

            if let Some(child) = child_opt {
                current = child;
                continue;
            }

            match self.tree.try_expand(&state, &mut self.bufs) {
                ExpandResult::NeedsEval(sign) => {
                    let state_key = state.state_key();
                    let d = self.bufs.path.len() as u32;
                    self.depth_max = self.depth_max.max(d);
                    self.sims_done += 1;
                    let path = self.bufs.take_path();
                    self.tree.apply_virtual_loss(&path);
                    return Select::Eval(
                        LeafId {
                            inner: LeafInner::Simulation {
                                sign,
                                actions: self.bufs.take_actions(),
                                path,
                                state_key,
                            },
                        },
                        state,
                    );
                }
                ExpandResult::Chance(id) => {
                    self.tree.set_child(current, edge_idx, id);
                    current = id;
                }
                ExpandResult::Leaf(id) => {
                    self.tree.set_child(current, edge_idx, id);
                    break;
                }
            }
        }

        // Synchronous backprop for terminal / transposition-hit leaves.
        self.tree.backprop(&self.bufs.path);
        let d = self.bufs.path.len() as u32;
        self.depth_max = self.depth_max.max(d);
        widen_q_bounds(&self.tree, &self.bufs.path, &mut self.q_bounds);
        self.sims_done += 1;

        // Check if the leaf is terminal for Select::Terminal return.
        let leaf_node = if let Some(&(_, eidx)) = self.bufs.path.last() {
            let parent = self.bufs.path.last().unwrap().0;
            self.tree.edges(parent)[eidx].child
        } else {
            self.root
        };
        if let Some(leaf_id) = leaf_node
            && matches!(*self.tree.kind(leaf_id), NodeKind::Terminal)
        {
            return Select::Terminal(
                LeafId {
                    inner: LeafInner::Terminal,
                },
                self.tree.wdl(leaf_id),
            );
        }

        // Not terminal — check if we should continue or are done.
        if self.sims_done >= self.config.num_simulations {
            Select::Done
        } else {
            // Recurse for the next sim.
            self.simulate(rng)
        }
    }
}

// ── Selection ─────────────────────────────────────────────────────────

/// Root selection: PUCT with Dirichlet noise.
fn puct_select(
    tree: &Tree,
    node_id: NodeId,
    sign: f32,
    config: &Config,
    legal_edges: &[usize],
    noise: Option<&[f32]>,
) -> usize {
    let edges = tree.edges(node_id);
    let total_visits: u32 = edges.iter().map(|e| e.visits).sum();
    let sqrt_total = (total_visits as f32).sqrt();
    let vmix_val = v_mix(tree, node_id);

    let mut best_idx = legal_edges[0];
    let mut best_score = f32::NEG_INFINITY;

    for &ei in legal_edges {
        let edge = &edges[ei];
        let prior = if let Some(n) = noise {
            (1.0 - config.dirichlet_epsilon) * edge.prior + config.dirichlet_epsilon * n[ei]
        } else {
            edge.prior
        };

        let cq = completed_q(tree, edge, vmix_val);
        // Q from current player's perspective (flip for minimizer)
        let q = cq * sign;
        let exploration = config.c_puct * prior * sqrt_total / (1.0 + edge.visits as f32);
        let score = q + exploration;

        if score > best_score {
            best_score = score;
            best_idx = ei;
        }
    }
    best_idx
}

/// Non-root selection: deterministic improved-policy selection.
/// π' = softmax(logit + σ(normalized_completedQ))
/// select argmax_a (π'(a) - N(a) / (1 + Σ N))
fn interior_select(
    tree: &Tree,
    node_id: NodeId,
    sign: f32,
    config: &Config,
    q_bounds: (f32, f32),
    scratch: &mut Vec<f32>,
    legal_edges: &[usize],
) -> usize {
    let edges = tree.edges(node_id);
    let total_child_visits: u32 = edges.iter().map(|e| e.visits).sum();
    let max_visits = tree.max_edge_visits(node_id);
    let vmix_val = v_mix(tree, node_id);

    // Improved policy logits for legal edges only
    scratch.clear();
    for &ei in legal_edges {
        let edge = &edges[ei];
        let cq = completed_q(tree, edge, vmix_val);
        let q_norm = normalize_q(cq, q_bounds.0, q_bounds.1, sign);
        let s = sigma(q_norm, max_visits, config.c_visit, config.c_scale);
        scratch.push(edge.logit + s);
    }
    softmax(scratch);

    // argmax(π'(a) - N(a) / (1 + Σ N)) over legal edges
    let denom = 1.0 + total_child_visits as f32;
    legal_edges
        .iter()
        .enumerate()
        .map(|(scratch_idx, &edge_idx)| {
            let score = scratch[scratch_idx] - edges[edge_idx].visits as f32 / denom;
            (edge_idx, score)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .expect("legal_edges should not be empty")
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Normalize Q into [0, 1] from the current player's perspective.
fn normalize_q(q: f32, q_min: f32, q_max: f32, sign: f32) -> f32 {
    let (q, lo, hi) = if sign > 0.0 {
        (q, q_min, q_max)
    } else {
        (-q, -q_max, -q_min)
    };
    let range = hi - lo;
    if range <= f32::EPSILON {
        return 0.5;
    }
    (q - lo) / range
}

fn sigma(q_norm: f32, max_visits: u32, c_visit: f32, c_scale: f32) -> f32 {
    (c_visit + max_visits as f32) * c_scale * q_norm
}

/// completedQ for an edge: child's Q if visited, v_mix otherwise.
fn completed_q(tree: &Tree, edge: &Edge, vmix_val: f32) -> f32 {
    match edge.child {
        Some(child) => tree.q(child),
        None => vmix_val,
    }
}

/// Mixed value approximation (Appendix D, Equation 33).
fn v_mix(tree: &Tree, id: NodeId) -> f32 {
    let edges = tree.edges(id);
    let n_total: f32 = edges.iter().map(|e| e.visits).sum::<u32>() as f32;
    if n_total == 0.0 {
        return tree.utility(id);
    }

    let v = tree.utility(id);
    let mut prior_weighted_q = 0.0f32;
    let mut visited_prior_sum = 0.0f32;
    for edge in edges {
        if let Some(child) = edge.child {
            prior_weighted_q += edge.prior * tree.q(child);
            visited_prior_sum += edge.prior;
        }
    }
    let weighted_q = if visited_prior_sum > 0.0 {
        prior_weighted_q / visited_prior_sum
    } else {
        0.0
    };

    (v + n_total * weighted_q) / (1.0 + n_total)
}

/// In-place softmax: transforms logits into probabilities.
fn softmax(buf: &mut [f32]) {
    let max = buf.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for v in buf.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in buf.iter_mut() {
        *v /= sum;
    }
}

/// Widen `(q_min, q_max)` from nodes along a backprop path.
fn widen_q_bounds(tree: &Tree, path: &[(NodeId, usize)], bounds: &mut (f32, f32)) {
    for &(nid, eidx) in path {
        let node_q = tree.q(nid);
        bounds.0 = bounds.0.min(node_q);
        bounds.1 = bounds.1.max(node_q);

        if let Some(child_id) = tree.edges(nid)[eidx].child {
            let child_q = tree.q(child_id);
            bounds.0 = bounds.0.min(child_q);
            bounds.1 = bounds.1.max(child_q);
        }
    }
}

/// Walk the most-visited path from `root`, counting edges.
fn compute_pv_depth(tree: &Tree, root: NodeId) -> u32 {
    let mut node = root;
    let mut depth = 0u32;
    loop {
        let best = tree
            .edges(node)
            .iter()
            .filter_map(|e| e.child.map(|c| (c, e.visits)))
            .max_by_key(|&(_, v)| v);
        match best {
            Some((child, _)) => {
                depth += 1;
                node = child;
            }
            None => break,
        }
    }
    depth
}

// ── Dirichlet noise ─────────────────────────────────────────────────

/// Sample from Gamma(alpha, 1) using Marsaglia and Tsang's method.
fn sample_gamma(alpha: f32, rng: &mut fastrand::Rng) -> f32 {
    if alpha < 1.0 {
        // For alpha < 1, use the relation: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        let g = sample_gamma(alpha + 1.0, rng);
        let u = rng.f32().max(f32::MIN_POSITIVE);
        return g * u.powf(1.0 / alpha);
    }

    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.f32().max(f32::MIN_POSITIVE);
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Standard normal via Box-Muller transform.
fn sample_standard_normal(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(f32::MIN_POSITIVE);
    let u2 = rng.f32();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Sample from a symmetric Dirichlet distribution with concentration `alpha`.
fn sample_dirichlet(alpha: f32, n: usize, rng: &mut fastrand::Rng) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let mut samples: Vec<f32> = (0..n).map(|_| sample_gamma(alpha, rng)).collect();
    let sum: f32 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    } else {
        // Fallback: uniform
        samples.fill(1.0 / n as f32);
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Evaluator, RolloutEvaluator};

    fn run_to_completion<G: Game>(
        search: &mut Search<G>,
        evaluator: &impl Evaluator<G>,
        rng: &mut fastrand::Rng,
    ) -> SearchResult {
        loop {
            match search.select(rng) {
                Select::Eval(leaf, ref state) => {
                    let eval = evaluator.evaluate(state, rng);
                    search.backup(leaf, eval);
                }
                Select::Terminal(leaf, wdl) => {
                    search.backup_terminal(leaf, wdl);
                }
                Select::Done => return search.result(),
            }
        }
    }

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
                Status::Decision(1.0)
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
        let evaluator = RolloutEvaluator::default();
        let config = Config {
            num_simulations: 500,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(TrivialGame::new(), config);
        let result = run_to_completion(&mut search, &evaluator, &mut rng);

        assert_eq!(
            result.selected_action, 0,
            "MCTS should select winning action 0"
        );
        assert!(
            result.policy[0] > result.policy[1],
            "improved policy should favor action 0: policy = {:?}",
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
                None => Status::Decision(1.0),
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
        let evaluator = RolloutEvaluator::default();
        let config = Config {
            num_simulations: 200,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(TwoStepGame::new(), config);
        let result = run_to_completion(&mut search, &evaluator, &mut rng);

        let action = result.selected_action;
        search.apply_action(action);

        // Tree pointer should have followed the child edge.
        assert!(search.root.is_some(), "root should follow child edge");

        // Second search reuses tree; compaction happens inside select.
        let result2 = run_to_completion(&mut search, &evaluator, &mut rng);

        let total: f32 = result2.policy.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "policy should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn state_machine_terminal_root() {
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(
            TrivialGame {
                done: true,
                chose_win: true,
            },
            Config::default(),
        );
        match search.select(&mut rng) {
            Select::Terminal(_leaf, wdl) => {
                assert_eq!(wdl, Wdl::from_value(1.0));
            }
            Select::Eval(..) => panic!("terminal root should not need eval"),
            Select::Done => panic!("terminal root should return Terminal, not Done"),
        }
    }

    #[test]
    fn improved_policy_sums_to_one() {
        let evaluator = RolloutEvaluator::default();
        let config = Config {
            num_simulations: 100,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(TrivialGame::new(), config);
        let result = run_to_completion(&mut search, &evaluator, &mut rng);

        let total: f32 = result.policy.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "improved policy should sum to ~1.0, got {total}"
        );
    }

    /// Game that transitions into a degenerate state: Decision but no
    /// legal actions (e.g. Catan's DevCardDraw with an exhausted pool).
    #[derive(Clone)]
    struct DeadEndGame {
        stuck: bool,
    }

    impl Game for DeadEndGame {
        const NUM_ACTIONS: usize = 2;

        fn status(&self) -> Status {
            // Not terminal, even when stuck.
            Status::Decision(1.0)
        }
        fn legal_actions(&self, buf: &mut Vec<usize>) {
            if !self.stuck {
                buf.push(0);
            }
            // When stuck: no legal actions AND no chance outcomes.
        }
        fn apply_action(&mut self, _action: usize) {
            self.stuck = true;
        }
    }

    #[test]
    fn degenerate_root_handled() {
        let evaluator = RolloutEvaluator::default();
        let config = Config {
            num_simulations: 10,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        // Degenerate root is treated as terminal draw by try_expand.
        let mut search = Search::new(DeadEndGame { stuck: true }, config);
        let result = run_to_completion(&mut search, &evaluator, &mut rng);
        // Should complete without panic, treated as draw.
        assert_eq!(result.wdl, Wdl::DRAW);
    }
}
