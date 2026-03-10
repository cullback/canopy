mod tree;

use crate::eval::Evaluation;
use crate::game::{Game, Status};
use crate::player::Player;

use tree::{Bufs, ExpandResult, NodeId, NodeKind, Tree};

// ── Public types ──────────────────────────────────────────────────────

/// Gumbel AlphaZero MCTS configuration.
#[derive(Clone)]
pub struct Config {
    pub num_simulations: u32,
    /// Number of actions sampled via Gumbel-Top-k at the root (m).
    pub num_sampled_actions: u32,
    /// σ scaling parameter (controls Q influence on improved policy).
    pub c_visit: f32,
    /// σ scaling parameter.
    pub c_scale: f32,
    /// Leaves to collect per batch before requesting evaluation (1 = no batching).
    pub leaf_batch_size: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            num_sampled_actions: 16,
            c_visit: 50.0,
            c_scale: 1.0,
            leaf_batch_size: 1,
        }
    }
}

/// Result of an MCTS search.
pub struct SearchResult {
    /// Improved policy over `[0, NUM_ACTIONS)` (training target).
    pub policy: Vec<f32>,
    /// Root value estimate from P1's perspective.
    pub value: f32,
    /// The action selected by Sequential Halving to play.
    pub selected_action: usize,
    /// Raw network value (P1 perspective) before search corrections.
    pub network_value: f32,
    /// (action, Q) pairs for visited root children.
    pub children_q: Vec<(usize, f32)>,
    /// Action with highest raw prior (network policy argmax).
    pub prior_top1_action: usize,
}

/// One step of the MCTS state machine.
pub enum Step<'a, G: Game> {
    /// Search needs evaluation of one or more leaf states.
    /// Evaluate each state and pass the results (in the same order) to the
    /// next [`Search::supply`] call.
    NeedsEval(&'a [G]),
    /// Search complete — improved policy, value, and selected action are ready.
    Done(SearchResult),
}

/// Gumbel AlphaZero MCTS, driven as a state machine.
///
/// The search tree persists across moves so that subtrees explored in earlier
/// searches can be reused (the tree is compacted, not rebuilt, on each new
/// search).  The public methods form a simple protocol:
///
/// 1. [`new`](Self::new) — construct with a root game state and config.
/// 2. [`step`](Self::step) — the only stepping function.  Call with an
///    empty slice to start a search; feed evaluations back on subsequent
///    calls.  Returns [`Step::NeedsEval`] or [`Step::Done`].
/// 3. [`pending_states`](Self::pending_states) — borrow the leaf states that
///    need evaluation (valid after `NeedsEval`).
/// 4. [`apply_action`](Self::apply_action) — mirror actions as they happen.
///
/// The caller never touches the tree directly; all interaction goes through
/// `Step`.
pub struct Search<G: Game> {
    tree: Tree,
    /// `None` initially or after `apply_action` walks through an unexpanded
    /// child.  The next `step` will discard the old tree and start fresh.
    root: Option<NodeId>,
    root_state: G,
    bufs: Bufs,
    config: Config,
    /// Gumbel Sequential Halving state — present for decision roots, `None`
    /// for chance roots.  When `Some`, simulation budget is tracked inside
    /// `GumbelState::budget_remaining` (per-candidate round-robin with
    /// halving).  When `None`, `vanilla_budget_remaining` is used instead
    /// as a simple countdown.
    gumbel: Option<GumbelState>,
    /// Simulation budget for chance roots (no Gumbel).  Ignored when
    /// `gumbel` is `Some`, since that state tracks its own budget.
    vanilla_budget_remaining: u32,
    /// Q bounds for vanilla (non-Gumbel) simulations, used to normalize Q
    /// in interior selection.  Reset each search, widened after each sim.
    vanilla_q_bounds: (f32, f32),
    /// Raw network value for the root (P1 perspective), captured on first
    /// expansion and reused in the `SearchResult`.
    root_network_value: f32,
    /// Whether a search is currently active (between first `step` and `Done`).
    search_active: bool,
    /// Leaf states awaiting evaluation.  Filled by simulation, consumed by
    /// `step`.  Reused across calls to avoid allocation.
    pending_states: Vec<G>,
    /// Matching contexts for `pending_states` (same length, same order).
    pending_contexts: Vec<Phase>,
}

// ── Internal types ────────────────────────────────────────────────────

enum Phase {
    ExpandingRoot {
        player: Player,
        actions: Vec<usize>,
    },
    Simulating {
        player: Player,
        actions: Vec<usize>,
        path: Vec<(NodeId, usize)>,
        state_key: Option<u64>,
    },
}

/// Gumbel Sequential Halving state for root action selection.
struct GumbelState {
    /// The player at the root node (needed for Q sign-flipping).
    root_player: Player,
    /// Pre-summed g(a) + logit(a) per root edge (for candidate scoring).
    gumbel_scores: Vec<f32>,
    /// Raw logits per root edge (needed for improved policy in extract_gumbel_result).
    root_logits: Vec<f32>,
    /// Edge indices alive in sequential halving.
    candidates: Vec<usize>,
    /// Current SH phase (0-indexed).
    phase: u32,
    /// ceil(log2(m)).
    total_phases: u32,
    /// Sims allocated per candidate this phase.
    sims_per_candidate: u32,
    /// Sims completed this phase (monotonic counter; candidate_idx and
    /// phase-complete are derived from this).
    sims_this_phase: u32,
    /// Min Q across tree (P1 perspective) for normalization.
    q_min: f32,
    /// Max Q across tree (P1 perspective) for normalization.
    q_max: f32,
    /// Simulation budget remaining (Gumbel decision roots only).
    /// Decremented each simulation; when exhausted the search finishes
    /// even if Sequential Halving hasn't reduced to one candidate.
    budget_remaining: u32,
}

impl GumbelState {
    /// Which candidate in the round-robin is currently being simulated.
    fn candidate_idx(&self) -> usize {
        debug_assert!(self.sims_per_candidate > 0);
        debug_assert!(!self.candidates.is_empty());
        (self.sims_this_phase / self.sims_per_candidate) as usize % self.candidates.len()
    }

    /// Whether all candidates have received their allocation this phase.
    fn phase_complete(&self) -> bool {
        self.sims_this_phase >= self.sims_per_candidate * self.candidates.len() as u32
    }
}

/// Borrow the simulation path from a Phase (empty for root expansion).
fn phase_path(phase: &Phase) -> &[(NodeId, usize)] {
    match phase {
        Phase::Simulating { path, .. } => path,
        Phase::ExpandingRoot { .. } => &[],
    }
}

enum SimResult<G: Game> {
    Complete,
    NeedsEval { state: G, context: Phase },
}

// ── Public API ────────────────────────────────────────────────────────

impl<G: Game> Search<G> {
    /// Create a new search with the given root game state and config.
    ///
    /// The tree starts empty; the first call to [`step`](Self::step) with
    /// an empty slice will expand from scratch.
    pub fn new(root_state: G, config: Config) -> Self {
        Self {
            tree: Tree::default(),
            root: None,
            root_state,
            bufs: Bufs::default(),
            config,
            gumbel: None,
            vanilla_budget_remaining: 0,
            vanilla_q_bounds: (0.0, 0.0),
            root_network_value: 0.0,
            search_active: false,
            pending_states: Vec::new(),
            pending_contexts: Vec::new(),
        }
    }

    /// Read access to the internal game state.
    pub fn state(&self) -> &G {
        &self.root_state
    }

    /// Update the simulation budget (takes effect on the next search).
    pub fn set_num_simulations(&mut self, n: u32) {
        self.config.num_simulations = n;
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
        self.search_active = false;
    }

    /// Feed evaluations and advance the search.
    ///
    /// On the first call (when no search is active), pass an empty slice to
    /// start a new search.  On subsequent calls, pass one [`Evaluation`] per
    /// state from the previous [`Step::NeedsEval`] (same order).
    ///
    /// Returns [`Step::NeedsEval`] with the leaf states that need evaluation,
    /// or [`Step::Done`] when the search is complete.
    pub fn step(&mut self, evals: &[Evaluation], rng: &mut fastrand::Rng) -> Step<'_, G> {
        if !self.search_active {
            return self.begin_search(rng);
        }
        self.integrate_evals(evals, rng);
        if self.gumbel.is_some() {
            self.run_simulations(rng)
        } else {
            self.run_vanilla_sims(rng)
        }
    }

    /// Start a new search: compact/clear tree, initialize gumbel state.
    fn begin_search(&mut self, rng: &mut fastrand::Rng) -> Step<'_, G> {
        self.search_active = true;
        self.gumbel = None;
        self.vanilla_budget_remaining = self.config.num_simulations;
        self.vanilla_q_bounds = (0.0, 0.0);

        // Terminal root requires no tree logic — immediate result.
        if let Status::Terminal(reward) = self.root_state.status() {
            self.search_active = false;
            return Step::Done(SearchResult {
                policy: vec![0.0; G::NUM_ACTIONS],
                value: reward,
                selected_action: 0,
                network_value: 0.0,
                children_q: vec![],
                prior_top1_action: 0,
            });
        }

        if let Some(old_root) = self.root {
            // Reusing tree: compact the graph
            let new_root = self.tree.compact(old_root);
            self.root = Some(new_root);

            let root_value = self.tree.utility(new_root);
            self.root_network_value = root_value;
            let root_player = match *self.tree.kind(new_root) {
                NodeKind::Decision(p) => p,
                _ => return self.run_vanilla_sims(rng),
            };

            self.gumbel = Some(init_gumbel(
                &self.tree,
                new_root,
                root_value,
                root_player,
                &self.config,
                rng,
            ));

            self.run_simulations(rng)
        } else {
            // Start fresh: clear nodes but preserve allocations
            self.tree.clear();

            match self.tree.try_expand(&self.root_state, &mut self.bufs) {
                ExpandResult::Leaf(_) => unreachable!("empty non-terminal tree"),
                ExpandResult::Chance(id) => {
                    self.root = Some(id);
                    self.root_network_value = self.tree.utility(id);
                    self.run_vanilla_sims(rng)
                }
                ExpandResult::NeedsEval(player) => {
                    self.pending_states.clear();
                    self.pending_contexts.clear();
                    self.pending_states.push(self.root_state.clone());
                    self.pending_contexts.push(Phase::ExpandingRoot {
                        player,
                        actions: self.bufs.take_actions(),
                    });
                    Step::NeedsEval(&self.pending_states)
                }
            }
        }
    }

    /// Integrate evaluations into the tree for previously pending leaves.
    fn integrate_evals(&mut self, evals: &[Evaluation], rng: &mut fastrand::Rng) {
        debug_assert_eq!(evals.len(), self.pending_contexts.len());
        for (eval, context) in evals.iter().zip(self.pending_contexts.drain(..)) {
            match context {
                Phase::ExpandingRoot { player, actions } => {
                    let state_key = self.root_state.state_key();
                    let root = self.tree.complete_expand(eval, &actions, player, state_key);
                    self.root = Some(root);
                    self.root_network_value = eval.value;

                    // Initialize Gumbel state for root
                    self.gumbel = Some(init_gumbel(
                        &self.tree,
                        root,
                        eval.value,
                        player,
                        &self.config,
                        rng,
                    ));
                    self.bufs.reclaim_actions(actions);
                }
                Phase::Simulating {
                    player,
                    actions,
                    path,
                    state_key,
                } => {
                    self.tree.remove_virtual_loss(&path);
                    let &(parent, edge_idx) = path.last().unwrap();
                    if self.tree.edges(parent)[edge_idx].child.is_none() {
                        let child = match state_key.and_then(|k| self.tree.lookup(k)) {
                            Some(existing) => existing,
                            None => self.tree.complete_expand(eval, &actions, player, state_key),
                        };
                        self.tree.set_child(parent, edge_idx, child);
                    }
                    self.tree.recompute_q(&path);
                    self.bufs.reclaim_path(path);
                    self.bufs.reclaim_actions(actions);
                }
            }
        }
        self.pending_states.clear();
    }

    /// The current root node. Panics if root is unset (only possible for
    /// terminal roots returned as Step::Done, or after advance with no child).
    fn root(&self) -> NodeId {
        self.root
            .expect("root accessed before initialization or after terminal")
    }

    fn run_simulations(&mut self, rng: &mut fastrand::Rng) -> Step<'_, G> {
        let root = self.root();
        let network_value = self.root_network_value;
        let gs = self
            .gumbel
            .as_mut()
            .expect("run_simulations called without gumbel state");

        // Single candidate remaining — no more simulations needed.
        if gs.candidates.len() <= 1 {
            self.search_active = false;
            return Step::Done(extract_gumbel_result::<G>(
                &self.tree,
                root,
                gs,
                &self.config,
                network_value,
            ));
        }

        self.pending_states.clear();
        self.pending_contexts.clear();
        loop {
            // Check if SH is complete (1 candidate left or budget exhausted)
            if gs.candidates.len() <= 1 || gs.budget_remaining == 0 {
                if !self.pending_states.is_empty() {
                    return Step::NeedsEval(&self.pending_states);
                }
                self.search_active = false;
                return Step::Done(extract_gumbel_result::<G>(
                    &self.tree,
                    root,
                    gs,
                    &self.config,
                    network_value,
                ));
            }

            let forced_edge = gs.candidates[gs.candidate_idx()];
            let q_bounds = (gs.q_min, gs.q_max);
            let config = &self.config;
            match simulate(
                &mut self.tree,
                root,
                &self.root_state,
                rng,
                &mut self.bufs,
                Some(forced_edge),
                |tree, node, player| gumbel_interior_select(tree, node, player, config, q_bounds),
            ) {
                SimResult::Complete => {
                    advance_round_robin(gs, &self.tree, &self.bufs.path, root, &self.config);
                }
                SimResult::NeedsEval { state, context } => {
                    // Advance round-robin before applying virtual loss so that
                    // update_q_bounds reads Q values not yet polluted by this
                    // simulation's virtual loss (q_min/q_max never contract,
                    // so feeding in artificially low values widens the range
                    // permanently).
                    advance_round_robin(gs, &self.tree, phase_path(&context), root, &self.config);
                    self.tree.apply_virtual_loss(phase_path(&context));
                    self.pending_states.push(state);
                    self.pending_contexts.push(context);
                    if self.pending_states.len() as u32 >= self.config.leaf_batch_size {
                        return Step::NeedsEval(&self.pending_states);
                    }
                }
            }
        }
    }

    /// Fallback for roots without Gumbel state (chance roots).
    fn run_vanilla_sims(&mut self, rng: &mut fastrand::Rng) -> Step<'_, G> {
        let root = self.root();
        let config = &self.config;
        let mut q_bounds = self.vanilla_q_bounds;
        self.pending_states.clear();
        self.pending_contexts.clear();
        while self.vanilla_budget_remaining > 0 {
            match simulate(
                &mut self.tree,
                root,
                &self.root_state,
                rng,
                &mut self.bufs,
                None,
                |tree, node, player| gumbel_interior_select(tree, node, player, config, q_bounds),
            ) {
                SimResult::Complete => {
                    widen_q_bounds(&self.tree, &self.bufs.path, &mut q_bounds);
                    self.vanilla_budget_remaining -= 1;
                }
                SimResult::NeedsEval { state, context } => {
                    // Widen bounds before virtual loss (same rationale as Gumbel path)
                    widen_q_bounds(&self.tree, phase_path(&context), &mut q_bounds);
                    self.tree.apply_virtual_loss(phase_path(&context));
                    self.vanilla_budget_remaining -= 1;
                    self.pending_states.push(state);
                    self.pending_contexts.push(context);
                    if self.pending_states.len() as u32 >= self.config.leaf_batch_size {
                        self.vanilla_q_bounds = q_bounds;
                        return Step::NeedsEval(&self.pending_states);
                    }
                }
            }
        }
        self.vanilla_q_bounds = q_bounds;
        if !self.pending_states.is_empty() {
            return Step::NeedsEval(&self.pending_states);
        }
        self.search_active = false;
        let network_value = self.root_network_value;
        Step::Done(visit_count_result::<G>(&self.tree, root, network_value))
    }
}

// ── Simulation ────────────────────────────────────────────────────────

fn simulate<G: Game>(
    tree: &mut Tree,
    root: NodeId,
    root_state: &G,
    rng: &mut fastrand::Rng,
    bufs: &mut Bufs,
    forced_root_edge: Option<usize>,
    select_decision: impl Fn(&Tree, NodeId, Player) -> usize,
) -> SimResult<G> {
    bufs.path.clear();
    let mut current = root;
    let mut state = root_state.clone();
    let mut forced = forced_root_edge;

    loop {
        let edge_idx = match *tree.kind(current) {
            NodeKind::Terminal => break,
            NodeKind::Chance => tree.sample_chance_edge(current, rng),
            NodeKind::Decision(player) => forced
                .take()
                .unwrap_or_else(|| select_decision(tree, current, player)),
        };

        let edges = tree.edges(current);
        bufs.path.push((current, edge_idx));
        let action = edges[edge_idx].action;
        let child_opt = edges[edge_idx].child;
        state.apply_action(action);

        if let Some(child) = child_opt {
            current = child;
            continue;
        }

        match tree.try_expand(&state, bufs) {
            ExpandResult::NeedsEval(player) => {
                let state_key = state.state_key();
                return SimResult::NeedsEval {
                    state,
                    context: Phase::Simulating {
                        player,
                        actions: bufs.take_actions(),
                        path: bufs.take_path(),
                        state_key,
                    },
                };
            }
            ExpandResult::Chance(id) => {
                tree.set_child(current, edge_idx, id);
                current = id;
            }
            ExpandResult::Leaf(id) => {
                tree.set_child(current, edge_idx, id);
                break;
            }
        }
    }

    tree.backprop(&bufs.path);
    SimResult::Complete
}

// ── Selection ─────────────────────────────────────────────────────────

/// Non-root selection: deterministic improved-policy selection.
/// π' = softmax(logit + σ(normalized_completedQ))
/// select argmax_a (π'(a) - N(a) / (1 + Σ N))
///
/// `q_bounds` is the global (q_min, q_max) from the tree, used for
/// consistent Q normalization across all nodes (per the paper).
///
/// Panics if the node has no edges (decision nodes must have legal actions).
fn gumbel_interior_select(
    tree: &Tree,
    node_id: NodeId,
    player: Player,
    config: &Config,
    q_bounds: (f32, f32),
) -> usize {
    let edges = tree.edges(node_id);
    let total_child_visits: u32 = edges.iter().map(|e| e.visits).sum();
    let max_visits = tree.max_edge_visits(node_id);
    let vmix_val = v_mix(tree, node_id);

    // Build improved policy logits: logit + σ(normalized Q)
    let mut improved_logits = Vec::with_capacity(edges.len());
    for edge in edges {
        let cq = completed_q(tree, edge, vmix_val);
        let q_norm = normalize_q_for_player(cq, q_bounds.0, q_bounds.1, player);
        let s = sigma(q_norm, max_visits, config.c_visit, config.c_scale);
        improved_logits.push(edge.logit + s);
    }

    // Softmax the improved logits
    let improved_policy = softmax(&improved_logits);

    // Select argmax(π'(a) - N(a) / (1 + Σ N))
    let denom = 1.0 + total_child_visits as f32;
    edges
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let score = improved_policy[i] - e.visits as f32 / denom;
            (i, score)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .expect("decision node should have edges")
}

// ── Gumbel helpers ────────────────────────────────────────────────────

fn normalize_q(q: f32, q_min: f32, q_max: f32) -> f32 {
    let range = q_max - q_min;
    if range <= f32::EPSILON {
        return 0.5;
    }
    (q - q_min) / range
}

/// Normalize Q from a player's perspective, flipping bounds for P2.
fn normalize_q_for_player(q: f32, q_min: f32, q_max: f32, player: Player) -> f32 {
    let sign = player.sign();
    let (pmin, pmax) = if sign > 0.0 {
        (q_min, q_max)
    } else {
        (-q_max, -q_min)
    };
    normalize_q(sign * q, pmin, pmax)
}

fn sigma(q_norm: f32, max_visits: u32, c_visit: f32, c_scale: f32) -> f32 {
    (c_visit + max_visits as f32) * c_scale * q_norm
}

/// completedQ for an edge: child's Q if visited, v_mix otherwise.
fn completed_q(tree: &Tree, edge: &tree::Edge, vmix_val: f32) -> f32 {
    match edge.child {
        Some(child) => tree.q(child),
        None => vmix_val,
    }
}

/// Mixed value approximation (Appendix D, Equation 33).
///
/// Approximates v_π = Σ_a π(a)·q(a), the expected value under the policy.
///
/// ```text
/// v_mix = (v̂ + N_total · Σ_{visited} π(a)·q(a) / Σ_{visited} π(b)) / (1 + N_total)
/// ```
///
/// **Why prior weights, not visit counts?**  We're estimating an expectation
/// under the *policy*, so the natural weights are the policy probabilities.
/// Visit counts reflect where the search happened to look (biased by
/// Gumbel's forced root edges), not what the policy believes.
///
/// **Why renormalize over visited edges only?**  We only know q(a) for
/// visited actions.  Dividing by the visited prior mass gives the
/// conditional expectation E_π[q | visited], the best estimate we have.
///
/// **Why interpolate with v̂?**  Early in search, few edges are visited and
/// the Q estimates are noisy, so we lean on the value network's prior.  As
/// N_total grows, the prior-weighted Q average becomes reliable and
/// dominates.
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
    // Conditional expectation of q under the policy, restricted to visited edges.
    let weighted_q = if visited_prior_sum > 0.0 {
        prior_weighted_q / visited_prior_sum
    } else {
        0.0
    };

    (v + n_total * weighted_q) / (1.0 + n_total)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    probs
}

/// Sample Gumbel(0) = -ln(-ln(U)).
fn sample_gumbel(rng: &mut fastrand::Rng) -> f32 {
    let u = 1.0 - rng.f32(); // avoid ln(0)
    -((-u.ln()).ln())
}

/// Initialize Gumbel state after root expansion.
fn init_gumbel(
    tree: &Tree,
    root: NodeId,
    root_value: f32,
    root_player: Player,
    config: &Config,
    rng: &mut fastrand::Rng,
) -> GumbelState {
    let edges = tree.edges(root);
    let num_edges = edges.len();

    let root_logits: Vec<f32> = edges.iter().map(|e| e.logit).collect();
    let gumbel_scores: Vec<f32> = root_logits
        .iter()
        .map(|&l| sample_gumbel(rng) + l)
        .collect();

    // Gumbel-Top-k: score = g + logit, take top m
    let m = (config.num_sampled_actions as usize)
        .min(num_edges)
        .min(config.num_simulations as usize);

    let mut scored: Vec<(usize, f32)> = gumbel_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    let candidates: Vec<usize> = scored.iter().take(m).map(|&(i, _)| i).collect();

    let m = candidates.len();
    let total_phases = if m <= 1 {
        1
    } else {
        (m as f32).log2().ceil() as u32
    };
    let budget = config.num_simulations;
    // The paper's Sequential Halving allocates floor(n / (ceil(log2(m)) * |S_k|))
    // per candidate per phase, where |S_k| shrinks each phase — so later phases
    // with fewer candidates get proportionally more sims each. Here we allocate
    // uniformly for phase 0; halve_candidates then redistributes the remaining
    // budget over fewer candidates, achieving the same escalation adaptively
    // while also reclaiming any rounding leftovers.
    let sims_per_candidate = if m > 0 && total_phases > 0 {
        (budget / (total_phases * m as u32)).max(1)
    } else {
        1
    };

    GumbelState {
        root_player,
        gumbel_scores,
        root_logits,
        candidates,
        phase: 0,
        total_phases,
        sims_per_candidate,
        sims_this_phase: 0,
        q_min: root_value,
        q_max: root_value,
        budget_remaining: budget,
    }
}

/// Precomputed root state for candidate scoring.
struct RootContext<'a> {
    tree: &'a Tree,
    edges: &'a [tree::Edge],
    vmix_val: f32,
    max_visits: u32,
}

impl<'a> RootContext<'a> {
    fn new(tree: &'a Tree, root: NodeId) -> Self {
        Self {
            edges: tree.edges(root),
            vmix_val: v_mix(tree, root),
            max_visits: tree.max_edge_visits(root),
            tree,
        }
    }
}

/// Score a candidate edge: g(a) + logit(a) + σ(completedQ(a)).
fn score_candidate(edge_idx: usize, gs: &GumbelState, ctx: &RootContext, config: &Config) -> f32 {
    let cq = completed_q(ctx.tree, &ctx.edges[edge_idx], ctx.vmix_val);
    let q_norm = normalize_q_for_player(cq, gs.q_min, gs.q_max, gs.root_player);
    let s = sigma(q_norm, ctx.max_visits, config.c_visit, config.c_scale);
    gs.gumbel_scores[edge_idx] + s
}

/// Halve candidates at end of a Sequential Halving phase.
fn halve_candidates(gs: &mut GumbelState, tree: &Tree, root: NodeId, config: &Config) {
    let ctx = RootContext::new(tree, root);

    // Score all candidates
    let mut scored: Vec<(usize, f32)> = gs
        .candidates
        .iter()
        .map(|&idx| (idx, score_candidate(idx, gs, &ctx, config)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Keep top half (ceil)
    let keep = scored.len().div_ceil(2);
    gs.candidates = scored.into_iter().take(keep).map(|(idx, _)| idx).collect();

    gs.phase += 1;
    gs.sims_this_phase = 0;

    // Recompute sims_per_candidate for remaining budget and candidates
    let remaining_phases = gs.total_phases.saturating_sub(gs.phase);
    let m = gs.candidates.len() as u32;
    if remaining_phases > 0 && m > 0 {
        gs.sims_per_candidate = (gs.budget_remaining / (remaining_phases * m)).max(1);
    } else {
        gs.sims_per_candidate = 1;
    }
}

/// Advance Sequential Halving round-robin after a simulation is committed.
///
/// Called at queue time for both synchronously completed simulations and
/// in-flight leaves (before eval returns).  Budget and phase transitions
/// proceed as if the simulation completed, even though the eval hasn't
/// returned yet.  This is a deliberate tradeoff:
///
/// **Why not defer to `step`?**  The batch collection loop needs the
/// round-robin counter to decide which candidate edge to force next.
/// Deferring would mean we can't advance the round-robin during
/// collection — we'd have to serialize leaf collection or invent
/// deferred bookkeeping to reconcile counters after evals arrive.
///
/// **Why halving at queue time is acceptable:**  Round-robin distributes
/// in-flight sims roughly evenly across candidates, so virtual-loss bias
/// is approximately symmetric and relative candidate ranking is preserved.
/// Q bounds are updated *before* virtual loss is applied (see call site),
/// preventing permanent widening from pessimistic values.  The
/// `debug_assert` in `run_simulations` guards against `leaf_batch_size`
/// dominating the phase budget, which would make most halving data
/// virtual-loss-polluted.
fn advance_round_robin(
    gs: &mut GumbelState,
    tree: &Tree,
    path: &[(NodeId, usize)],
    root: NodeId,
    config: &Config,
) {
    update_q_bounds(gs, tree, path);
    gs.budget_remaining = gs.budget_remaining.saturating_sub(1);
    gs.sims_this_phase += 1;

    if gs.phase_complete() {
        halve_candidates(gs, tree, root, config);
    }
}

/// Update q_min/q_max from nodes touched in the backprop path.
///
/// Bounds only ever widen (never contract), and only walk the path
/// (O(depth)), not the full tree (O(nodes)).  This means bounds are
/// underestimates of the true range early in search, which compresses σ
/// and makes selection more policy-driven.  As more of the tree is
/// visited the bounds widen and Q gets more influence — a reasonable
/// warmup: trust the policy when search data is sparse, trust Q as it
/// becomes reliable.
///
/// For in-flight leaves, this is called *before* `apply_virtual_loss` to
/// avoid permanently widening bounds with artificially pessimistic Q.
fn update_q_bounds(gs: &mut GumbelState, tree: &Tree, path: &[(NodeId, usize)]) {
    let mut bounds = (gs.q_min, gs.q_max);
    widen_q_bounds(tree, path, &mut bounds);
    gs.q_min = bounds.0;
    gs.q_max = bounds.1;
}

/// Widen `(q_min, q_max)` from nodes along a backprop path.
/// Same logic as `update_q_bounds` but operates on a bare tuple instead of
/// `GumbelState`, for use in vanilla (non-Gumbel) simulations.
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

/// Build a result from visit counts (used for chance roots without Gumbel state).
fn visit_count_result<G: Game>(tree: &Tree, root: NodeId, network_value: f32) -> SearchResult {
    let edges = tree.edges(root);
    let total_visits: u32 = edges.iter().map(|e| e.visits).sum();
    let mut policy = vec![0.0f32; G::NUM_ACTIONS];
    let mut best_action = 0;
    let mut best_visits = 0;
    if total_visits > 0 {
        for edge in edges {
            policy[edge.action] = edge.visits as f32 / total_visits as f32;
            if edge.visits > best_visits {
                best_visits = edge.visits;
                best_action = edge.action;
            }
        }
    }
    let prior_top1_action = edges
        .iter()
        .max_by(|a, b| a.logit.total_cmp(&b.logit))
        .map(|e| e.action)
        .unwrap_or(0);
    let children_q: Vec<(usize, f32)> = edges
        .iter()
        .filter_map(|e| e.child.map(|c| (e.action, tree.q(c))))
        .collect();
    SearchResult {
        policy,
        value: tree.q(root),
        selected_action: best_action,
        network_value,
        children_q,
        prior_top1_action,
    }
}

/// Extract Gumbel search result.
fn extract_gumbel_result<G: Game>(
    tree: &Tree,
    root: NodeId,
    gs: &GumbelState,
    config: &Config,
    network_value: f32,
) -> SearchResult {
    let ctx = RootContext::new(tree, root);

    // selected_action: argmax over final candidates of g + logit + σ(completedQ)
    let selected_edge = gs
        .candidates
        .iter()
        .map(|&idx| (idx, score_candidate(idx, gs, &ctx, config)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, _)| idx)
        .expect("candidates should not be empty");
    let selected_action = ctx.edges[selected_edge].action;

    // Improved policy (training target): softmax(logit + σ(completedQ)) over ALL edges.
    // Unvisited edges use v_mix as their completedQ estimate (see completed_q).
    // This is the paper's approach but can be noisy when few edges are visited,
    // since v_mix interpolates the value network prior with a sparse Q average.
    let mut improved_logits = Vec::with_capacity(ctx.edges.len());
    for (i, edge) in ctx.edges.iter().enumerate() {
        let cq = completed_q(ctx.tree, edge, ctx.vmix_val);
        let q_norm = normalize_q_for_player(cq, gs.q_min, gs.q_max, gs.root_player);
        let s = sigma(q_norm, ctx.max_visits, config.c_visit, config.c_scale);
        improved_logits.push(gs.root_logits[i] + s);
    }
    let improved_probs = softmax(&improved_logits);

    let mut policy = vec![0.0f32; G::NUM_ACTIONS];
    for (edge, &prob) in ctx.edges.iter().zip(&improved_probs) {
        policy[edge.action] = prob;
    }

    // Network's top-1 action (highest raw prior logit)
    let prior_top1_action = gs
        .root_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| ctx.edges[i].action)
        .unwrap_or(selected_action);

    // Q values for visited root children
    let children_q: Vec<(usize, f32)> = ctx
        .edges
        .iter()
        .filter_map(|e| e.child.map(|c| (e.action, tree.q(c))))
        .collect();

    SearchResult {
        policy,
        value: tree.q(root),
        selected_action,
        network_value,
        children_q,
        prior_top1_action,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Evaluator, RolloutEvaluator};
    use crate::player::Player;

    fn run_to_completion<G: Game>(
        search: &mut Search<G>,
        evaluator: &impl Evaluator<G>,
        rng: &mut fastrand::Rng,
    ) -> SearchResult {
        let mut evals = vec![];
        loop {
            match search.step(&evals, rng) {
                Step::NeedsEval(states) => {
                    let refs: Vec<&G> = states.iter().collect();
                    evals = evaluator.evaluate_batch(&refs, rng);
                }
                Step::Done(result) => return result,
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
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
        let config = Config {
            num_simulations: 500,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(TrivialGame::new(), config);
        let result = run_to_completion(&mut search, &evaluator, &mut rng);

        assert_eq!(
            result.selected_action, 0,
            "Gumbel MCTS should select winning action 0"
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

        let mut search = Search::new(TwoStepGame::new(), config);
        let result = run_to_completion(&mut search, &evaluator, &mut rng);

        let action = result.selected_action;
        let node_count_before = search.tree.node_count();
        search.apply_action(action);

        // Second search reuses tree; compaction happens inside supply.
        let result2 = run_to_completion(&mut search, &evaluator, &mut rng);
        let node_count_after_compact = search.tree.node_count();
        assert!(
            node_count_after_compact < node_count_before,
            "retain_subtree should compact: {node_count_after_compact} >= {node_count_before}"
        );

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
        match search.step(&[], &mut rng) {
            Step::Done(result) => {
                assert_eq!(result.value, 1.0);
                assert!(result.policy.iter().all(|&p| p == 0.0));
            }
            Step::NeedsEval(_) => panic!("terminal root should not need eval"),
        }
    }

    #[test]
    fn improved_policy_sums_to_one() {
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
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

    #[test]
    fn sequential_halving_halves_candidates() {
        // With 4 candidates, after one phase we should have 2
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
        let config = Config {
            num_simulations: 100,
            num_sampled_actions: 4, // but TrivialGame only has 2 actions
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(TrivialGame::new(), config);
        let result = run_to_completion(&mut search, &evaluator, &mut rng);

        // Should complete successfully and pick the winning action
        assert_eq!(result.selected_action, 0);
    }
}
