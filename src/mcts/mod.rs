mod seq_halving;
mod tree;

use crate::eval::{Evaluation, wdl_from_scalar};
use crate::game::{Game, Status};

use seq_halving::Schedule;
use tree::{Bufs, ExpandResult, NodeId, NodeKind, Tree};

// ── Snapshot types ───────────────────────────────────────────────────

/// Snapshot of a single root edge (action) for the analysis board.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct EdgeSnapshot {
    pub action: usize,
    pub visits: u32,
    /// Q value for this edge's child (None if unvisited).
    pub q: Option<f32>,
    /// Softmax prior probability.
    pub prior: f32,
    /// Raw policy logit.
    pub logit: f32,
    /// Improved policy weight (None if no gumbel state).
    pub improved_policy: Option<f32>,
}

/// Snapshot of the root search state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SearchSnapshot {
    /// Root WDL (P1 perspective).
    pub root_wdl: [f32; 3],
    /// Raw network value at root (P1 perspective).
    pub network_value: f32,
    /// Total simulations completed so far.
    pub total_simulations: u32,
    /// Per-edge snapshots, sorted by edge index.
    pub edges: Vec<EdgeSnapshot>,
}

/// Recursive snapshot of a subtree node for the tree explorer.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TreeNodeSnapshot {
    /// Action that led to this node (None for root).
    pub action: Option<usize>,
    /// Label for this node (game-specific, set by caller).
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub label: Option<String>,
    /// WDL of this node (P1 perspective).
    pub wdl: [f32; 3],
    /// Total visits.
    pub visits: u32,
    /// Node kind label: "decision", "chance", or "terminal".
    pub kind: &'static str,
    /// Which player acts at this node: 0 = P1, 1 = P2, None = chance/terminal.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub player: Option<u8>,
    /// Children (empty if leaf or max_depth reached).
    pub children: Vec<TreeNodeSnapshot>,
}

// ── Public types ──────────────────────────────────────────────────────

/// Gumbel AlphaZero MCTS configuration.
#[derive(Clone)]
pub struct Config {
    /// Simulation budget per search. Set to 0 to skip search entirely and
    /// select an action from the evaluator's policy alone (useful for
    /// heuristic evaluators that encode a fixed strategy).
    pub num_simulations: u32,
    /// Number of actions sampled via Gumbel-Top-k at the root (m).
    /// Automatically clamped to the number of legal actions, so values
    /// larger than `NUM_ACTIONS` are safe and behave identically.
    pub num_sampled_actions: u32,
    /// σ scaling parameter (controls Q influence on improved policy).
    pub c_visit: f32,
    /// σ scaling parameter.
    pub c_scale: f32,
    /// Leaves to collect per batch before requesting evaluation (1 = no batching).
    pub leaf_batch_size: u32,
    /// Scale of the Gumbel noise at the root. Use 1.0 (default) for
    /// stochastic or imperfect-information games; 0.0 for deterministic
    /// perfect-information games where exploration noise is unnecessary.
    pub gumbel_scale: f32,
    /// When true, interior decision nodes filter tree edges against the
    /// determinized state's legal actions (SO-ISMCTS). Required for games
    /// with hidden information where redeterminization can make stored
    /// edges illegal.
    pub filter_legal: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            num_sampled_actions: 4,
            c_visit: 50.0,
            c_scale: 1.0,
            leaf_batch_size: 1,
            gumbel_scale: 1.0,
            filter_legal: false,
        }
    }
}

/// Result of an MCTS search.
pub struct SearchResult {
    /// Improved policy over `[0, NUM_ACTIONS)` (training target).
    pub policy: Vec<f32>,
    /// Root WDL (P1 perspective).
    pub wdl: [f32; 3],
    /// The action selected by Sequential Halving to play.
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
/// The search tree persists across actions so that subtrees explored in earlier
/// searches can be reused (the tree is compacted, not rebuilt, on each new
/// search).  The public methods form a simple protocol:
///
/// 1. [`new`](Self::new) — construct with a root game state and config.
/// 2. [`step`](Self::step) — the only stepping function.  Call with an
///    empty slice to start a search; feed evaluations back on subsequent
///    calls.  Returns [`Step::NeedsEval`] or [`Step::Done`].
/// 3. [`apply_action`](Self::apply_action) — mirror actions as they happen.
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
    /// for chance roots.  When `Some`, the pre-computed schedule drives
    /// simulation assignment and halving.  When `None`,
    /// `vanilla_budget_remaining` is used instead as a simple countdown.
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
    /// Maximum simulation depth across all sims in the current search.
    depth_max: u32,
    /// Leaf states awaiting evaluation.  Filled by simulation, consumed by
    /// `step`.  Reused across calls to avoid allocation.
    pending_states: Vec<G>,
    /// Matching contexts for `pending_states` (same length, same order).
    pending_contexts: Vec<Phase>,
}

// ── Internal types ────────────────────────────────────────────────────

enum Phase {
    ExpandingRoot {
        sign: f32,
        actions: Vec<usize>,
    },
    Simulating {
        sign: f32,
        actions: Vec<usize>,
        path: Vec<(NodeId, usize)>,
        state_key: Option<u64>,
    },
}

/// Gumbel Sequential Halving state for root action selection.
struct GumbelState {
    /// The sign at the root node (needed for Q sign-flipping).
    root_sign: f32,
    /// Pre-summed g(a) + logit(a) per root edge (for candidate scoring).
    gumbel_scores: Vec<f32>,
    /// Raw logits per root edge (needed for improved policy in extract_gumbel_result).
    root_logits: Vec<f32>,
    /// Edge indices alive in sequential halving.
    candidates: Vec<usize>,
    /// Pre-computed Sequential Halving schedule (candidate offsets + halving points).
    schedule: Schedule,
    /// Monotonic simulation counter, incremented at queue time.
    sim_index: usize,
    /// Min Q across tree (P1 perspective) for normalization.
    q_min: f32,
    /// Max Q across tree (P1 perspective) for normalization.
    q_max: f32,
    /// Root edge indices that are legal in the actual game state.
    /// `None` when all edges are legal (fresh expansion or no filtering).
    legal_edges: Option<Vec<usize>>,
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
            depth_max: 0,
            pending_states: Vec::new(),
            pending_contexts: Vec::new(),
        }
    }

    /// Reset for a new game, reusing internal allocations.
    pub fn reset(&mut self, root_state: G) {
        self.tree.clear();
        self.root = None;
        self.root_state = root_state;
        self.gumbel = None;
        self.vanilla_budget_remaining = 0;
        self.vanilla_q_bounds = (0.0, 0.0);
        self.root_network_value = 0.0;
        self.search_active = false;
        self.pending_states.clear();
        self.pending_contexts.clear();
    }

    /// Read access to the internal game state.
    pub fn state(&self) -> &G {
        &self.root_state
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
    /// Safe to call even when no search is active (no-op). Use when a
    /// previous search may have been interrupted (e.g. by a panic in the
    /// evaluator or a dropped connection) and the caller wants to start
    /// fresh without stale `pending_contexts` triggering assertion failures.
    pub fn cancel_search(&mut self) {
        for context in self.pending_contexts.drain(..) {
            if let Phase::Simulating { ref path, .. } = context {
                self.tree.remove_virtual_loss(path);
            }
        }
        self.pending_states.clear();
        self.search_active = false;
    }

    /// Update the simulation budget (takes effect on the next search).
    /// Clears the Gumbel state so `begin_search` creates a fresh schedule
    /// for the new budget. The tree is preserved for reuse.
    pub fn set_num_simulations(&mut self, n: u32) {
        self.config.num_simulations = n;
        self.gumbel = None;
    }

    /// Whether a search is currently active (between first `step` and `Done`).
    pub fn is_searching(&self) -> bool {
        self.search_active
    }

    /// Enable or disable SO-ISMCTS legal-action filtering at interior nodes.
    pub fn set_filter_legal(&mut self, enabled: bool) {
        self.config.filter_legal = enabled;
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

    /// Snapshot the current root search state for the analysis board.
    ///
    /// Returns `None` if the root is not yet expanded (no tree data).
    pub fn snapshot(&self) -> Option<SearchSnapshot> {
        let root = self.root?;
        let edges = self.tree.edges(root);

        // When filter_legal is on, the tree may have stale edges from
        // determinized simulations or a previous state. Filter to only
        // edges that are currently legal so the UI never shows ghost actions.
        let legal: Option<Vec<usize>> = if self.config.filter_legal {
            let mut buf = Vec::new();
            self.root_state.legal_actions(&mut buf);
            Some(buf)
        } else {
            None
        };

        // Compute improved policy if gumbel state is available and still
        // matches the current root edges (walk_tree / state changes can
        // invalidate the stored logits).
        let improved = self
            .gumbel
            .as_ref()
            .filter(|gs| gs.root_logits.len() == edges.len())
            .map(|gs| {
                let ctx = RootContext::new(&self.tree, root);
                let mut logits = Vec::with_capacity(edges.len());
                for (i, edge) in ctx.edges.iter().enumerate() {
                    let cq = completed_q(ctx.tree, edge, ctx.vmix_val);
                    let q_norm = normalize_q(cq, gs.q_min, gs.q_max, gs.root_sign);
                    let s = sigma(
                        q_norm,
                        ctx.max_visits,
                        self.config.c_visit,
                        self.config.c_scale,
                    );
                    logits.push(gs.root_logits[i] + s);
                }
                softmax(&mut logits);
                logits
            });

        let edge_snapshots: Vec<EdgeSnapshot> = edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| legal.as_ref().is_none_or(|l| l.contains(&edge.action)))
            .map(|(i, edge)| EdgeSnapshot {
                action: edge.action,
                visits: edge.visits,
                q: edge.child.map(|c| self.tree.q(c)),
                prior: edge.prior,
                logit: edge.logit,
                improved_policy: improved.as_ref().map(|ip| ip[i]),
            })
            .collect();

        let total_simulations: u32 = edge_snapshots.iter().map(|e| e.visits).sum();

        Some(SearchSnapshot {
            root_wdl: self.tree.wdl(root),
            network_value: self.root_network_value,
            total_simulations,
            edges: edge_snapshots,
        })
    }

    /// Snapshot the subtree rooted at the current root, up to `max_depth`.
    pub fn snapshot_subtree(&self, max_depth: usize) -> Option<TreeNodeSnapshot> {
        let root = self.root?;
        Some(snapshot_node(&self.tree, root, None, None, max_depth))
    }

    /// Follow a path of actions from root, then snapshot that subtree.
    ///
    /// Returns `None` if the root is not expanded or the path leads to
    /// an unexpanded node.
    pub fn snapshot_at_path(
        &self,
        actions: &[usize],
        max_depth: usize,
    ) -> Option<TreeNodeSnapshot> {
        let mut current = self.root?;
        let mut parent_player = None;
        for &action in actions {
            // The player at `current` is the one who chose `action`.
            if let NodeKind::Decision(sign) = self.tree.kind(current) {
                parent_player = Some(if *sign > 0.0 { 0u8 } else { 1 });
            }
            current = self.tree.child_for_action(current, action)?;
        }
        Some(snapshot_node(
            &self.tree,
            current,
            actions.last().copied(),
            parent_player,
            max_depth,
        ))
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
        self.gumbel = None;
    }

    /// Walk the tree pointer through a sequence of actions without touching
    /// `root_state`.
    ///
    /// Use when the caller will set `root_state` separately (e.g. via
    /// `update_state`). At chance nodes, falls back to the most-visited
    /// child if the exact outcome wasn't explored or no action was
    /// recorded for it. Stops early once the pointer becomes `None`.
    pub fn walk_tree(&mut self, actions: &[usize]) -> usize {
        let orig_root = self.root;
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
        self.search_active = false;
        // If the root moved, the existing Gumbel state (candidates, logits,
        // schedule) refers to a different node and must be discarded. Without
        // this, `begin_search`'s "resume existing gumbel" branch would reuse
        // a stale schedule — often already exhausted — causing a budget
        // refill to run zero new sims after an opponent move.
        if self.root != orig_root {
            self.gumbel = None;
        }
        walked
    }

    /// Principal variation depth: follow the most-visited edge from root,
    /// counting edges until reaching a node with no expanded children.
    fn pv_depth(&self) -> u32 {
        match self.root {
            Some(root) => compute_pv_depth(&self.tree, root),
            None => 0,
        }
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
    /// If the root hasn't changed and a previous Gumbel state exists,
    /// resumes the existing search with additional budget.
    fn begin_search(&mut self, rng: &mut fastrand::Rng) -> Step<'_, G> {
        self.search_active = true;
        self.vanilla_budget_remaining = self.config.num_simulations;
        self.vanilla_q_bounds = (0.0, 0.0);
        self.depth_max = 0;

        // Terminal root requires no tree logic — immediate result.
        if let Status::Terminal(reward) = self.root_state.status() {
            self.search_active = false;
            self.gumbel = None;
            return Step::Done(SearchResult {
                policy: vec![0.0; G::NUM_ACTIONS],
                wdl: wdl_from_scalar(reward),
                selected_action: 0,
                network_value: 0.0,
                children_q: vec![],
                prior_top1_action: 0,
                pv_depth: 0,
                max_depth: 0,
            });
        }

        if let Some(old_root) = self.root {
            // Reusing tree: compact the graph
            let new_root = self.tree.compact(old_root);
            self.root = Some(new_root);

            self.root_network_value = self.tree.utility(new_root);
            let root_value = self.root_network_value;
            let root_sign = match *self.tree.kind(new_root) {
                NodeKind::Decision(sign) => sign,
                NodeKind::Chance => {
                    self.gumbel = None;
                    return self.run_vanilla_sims(rng);
                }
                NodeKind::Terminal => {
                    // Stale node from a degenerate simulation (SO-ISMCTS
                    // state inconsistency). State is ongoing but the tree
                    // thinks it's terminal. Discard and start fresh.
                    self.root = None;
                    self.gumbel = None;
                    return self.begin_search(rng);
                }
            };

            // When filter_legal is on, tree-reused root edges may include
            // actions from a determinized simulation that are illegal in the
            // actual state. Filter candidates to only legal edges.
            let legal = if self.config.filter_legal {
                self.bufs.legal.clear();
                self.root_state.legal_actions(&mut self.bufs.legal);
                let edges = self.tree.edges(new_root);
                let legal_indices: Vec<usize> = (0..edges.len())
                    .filter(|&i| self.bufs.legal.contains(&edges[i].action))
                    .collect();
                // If no tree edges are legal, the tree is stale — discard and
                // start fresh rather than panicking on empty candidates.
                if legal_indices.is_empty() {
                    self.root = None;
                    self.gumbel = None;
                    return self.begin_search(rng);
                }
                Some(legal_indices)
            } else {
                None
            };

            // Single legal action — Gumbel can't halve, use vanilla sims
            // for WDL estimation only; the action is predetermined.
            let num_legal = legal
                .as_ref()
                .map_or(self.tree.edges(new_root).len(), |l| l.len());
            if num_legal <= 1 {
                self.gumbel = None;
                return self.run_vanilla_sims(rng);
            }

            // Resume existing Gumbel state if available (same root, same
            // candidates). This preserves sequential halving progress across
            // cancel/restart cycles caused by polling.
            if self.gumbel.is_none() {
                self.gumbel = Some(init_gumbel(
                    &self.tree,
                    new_root,
                    root_value,
                    root_sign,
                    &self.config,
                    rng,
                    legal,
                ));
            }

            self.run_simulations(rng)
        } else {
            // Start fresh: clear nodes but preserve allocations
            self.tree.clear();

            match self.tree.try_expand(&self.root_state, &mut self.bufs) {
                ExpandResult::Leaf(_) => {
                    // Degenerate non-terminal state: no chance outcomes and
                    // no legal actions. Log details for diagnosis — this
                    // indicates a game logic bug.
                    let mut actions = Vec::new();
                    self.root_state.legal_actions(&mut actions);
                    let mut chances = Vec::new();
                    self.root_state.chance_outcomes(&mut chances);
                    panic!(
                        "degenerate non-terminal root: status={:?}, \
                         legal_actions={actions:?}, chance_outcomes={chances:?}",
                        self.root_state.status(),
                    );
                }
                ExpandResult::Chance(id) => {
                    self.root = Some(id);
                    self.root_network_value = self.tree.utility(id);
                    self.run_vanilla_sims(rng)
                }
                ExpandResult::NeedsEval(sign) => {
                    self.pending_states.clear();
                    self.pending_contexts.clear();
                    self.pending_states.push(self.root_state.clone());
                    self.pending_contexts.push(Phase::ExpandingRoot {
                        sign,
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
                Phase::ExpandingRoot { sign, actions } => {
                    let state_key = self.root_state.state_key();
                    let root = self.tree.complete_expand(eval, &actions, sign, state_key);
                    self.root = Some(root);
                    self.root_network_value = eval.wdl[0] - eval.wdl[2];

                    // Initialize Gumbel state for root (fresh expansion: all edges legal).
                    // Skip Gumbel for single-action roots — vanilla sims suffice for WDL.
                    if actions.len() > 1 {
                        self.gumbel = Some(init_gumbel(
                            &self.tree,
                            root,
                            self.root_network_value,
                            sign,
                            &self.config,
                            rng,
                            None,
                        ));
                    }
                    self.bufs.reclaim_actions(actions);
                }
                Phase::Simulating {
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
                            None => self.tree.complete_expand(eval, &actions, sign, state_key),
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

    fn run_simulations(&mut self, rng: &mut fastrand::Rng) -> Step<'_, G> {
        let Some(root) = self.root else {
            self.search_active = false;
            return self.begin_search(rng);
        };
        let network_value = self.root_network_value;
        let gs = self
            .gumbel
            .as_mut()
            .expect("run_simulations called without gumbel state");

        self.pending_states.clear();
        self.pending_contexts.clear();
        loop {
            // SH complete: schedule exhausted or 1 candidate left.
            if gs.sim_index >= gs.schedule.len() || gs.candidates.len() <= 1 {
                if !self.pending_states.is_empty() {
                    return Step::NeedsEval(&self.pending_states);
                }
                self.search_active = false;
                let pv_depth = compute_pv_depth(&self.tree, root);
                let result = extract_gumbel_result::<G>(
                    &self.tree,
                    root,
                    gs,
                    &self.config,
                    network_value,
                    pv_depth,
                    self.depth_max,
                );
                return Step::Done(result);
            }

            let offset = gs.schedule.candidate_offset(gs.sim_index);
            let forced_edge = gs.candidates[offset % gs.candidates.len()];
            let q_bounds = (gs.q_min, gs.q_max);
            let config = &self.config;
            let mut scratch = std::mem::take(&mut self.bufs.scratch);
            let result = simulate(
                &mut self.tree,
                root,
                &self.root_state,
                rng,
                &mut self.bufs,
                Some(forced_edge),
                config.filter_legal,
                |tree, node, sign, legal| {
                    gumbel_interior_select(tree, node, sign, config, q_bounds, &mut scratch, legal)
                },
            );
            self.bufs.scratch = scratch;
            match result {
                SimResult::Complete => {
                    let d = self.bufs.path.len() as u32;
                    self.depth_max = self.depth_max.max(d);
                    advance_sim(gs, &self.tree, &self.bufs.path, root, &self.config);
                }
                SimResult::NeedsEval { state, context } => {
                    let d = phase_path(&context).len() as u32;
                    self.depth_max = self.depth_max.max(d);
                    // Advance before applying virtual loss so that
                    // update_q_bounds reads Q values not yet polluted by this
                    // simulation's virtual loss (q_min/q_max never contract,
                    // so feeding in artificially low values widens the range
                    // permanently).
                    advance_sim(gs, &self.tree, phase_path(&context), root, &self.config);
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
        let Some(root) = self.root else {
            self.search_active = false;
            return self.begin_search(rng);
        };
        let config = &self.config;
        let mut scratch = std::mem::take(&mut self.bufs.scratch);
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
                config.filter_legal,
                |tree, node, sign, legal| {
                    gumbel_interior_select(tree, node, sign, config, q_bounds, &mut scratch, legal)
                },
            ) {
                SimResult::Complete => {
                    let d = self.bufs.path.len() as u32;
                    self.depth_max = self.depth_max.max(d);
                    widen_q_bounds(&self.tree, &self.bufs.path, &mut q_bounds);
                    self.vanilla_budget_remaining -= 1;
                }
                SimResult::NeedsEval { state, context } => {
                    let d = phase_path(&context).len() as u32;
                    self.depth_max = self.depth_max.max(d);
                    // Widen bounds before virtual loss (same rationale as Gumbel path)
                    widen_q_bounds(&self.tree, phase_path(&context), &mut q_bounds);
                    self.tree.apply_virtual_loss(phase_path(&context));
                    self.vanilla_budget_remaining -= 1;
                    self.pending_states.push(state);
                    self.pending_contexts.push(context);
                    if self.pending_states.len() as u32 >= self.config.leaf_batch_size {
                        self.vanilla_q_bounds = q_bounds;
                        self.bufs.scratch = scratch;
                        return Step::NeedsEval(&self.pending_states);
                    }
                }
            }
        }
        self.vanilla_q_bounds = q_bounds;
        self.bufs.scratch = scratch;
        if !self.pending_states.is_empty() {
            return Step::NeedsEval(&self.pending_states);
        }
        self.search_active = false;
        let network_value = self.root_network_value;
        let pv_depth = self.pv_depth();
        // Pass legal actions so visit_count_result filters stale edges.
        let legal = if self.config.filter_legal {
            self.bufs.legal.clear();
            self.root_state.legal_actions(&mut self.bufs.legal);
            Some(self.bufs.legal.as_slice())
        } else {
            None
        };
        let result = visit_count_result::<G>(
            &self.tree,
            root,
            legal,
            network_value,
            pv_depth,
            self.depth_max,
        );
        Step::Done(result)
    }
}

// ── Simulation ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn simulate<G: Game>(
    tree: &mut Tree,
    root: NodeId,
    root_state: &G,
    rng: &mut fastrand::Rng,
    bufs: &mut Bufs,
    forced_root_edge: Option<usize>,
    filter_legal: bool,
    mut select_decision: impl FnMut(&Tree, NodeId, f32, &[usize]) -> usize,
) -> SimResult<G> {
    bufs.path.clear();
    let mut current = root;
    let mut state = root_state.clone();
    state.determinize(rng);
    let mut forced = forced_root_edge;

    loop {
        let edge_idx = match *tree.kind(current) {
            NodeKind::Terminal => break,
            NodeKind::Chance => {
                if filter_legal {
                    // SO-ISMCTS: resample from the determinized state so the
                    // outcome is consistent with the current information set.
                    // If the outcome has no matching tree edge (pool composition
                    // diverged from expansion time), abort this simulation.
                    match state.sample_chance(rng) {
                        Some(outcome) => {
                            let edges = tree.edges(current);
                            match edges.iter().position(|e| e.action == outcome) {
                                Some(idx) => idx,
                                None => break,
                            }
                        }
                        None => break,
                    }
                } else {
                    tree.sample_chance_edge(current, rng)
                }
            }
            NodeKind::Decision(sign) => {
                // Use forced root edge if legal; otherwise fall through to
                // normal selection so the simulation isn't wasted.
                if let Some(f) = forced.take() {
                    if !filter_legal || {
                        bufs.legal.clear();
                        state.legal_actions(&mut bufs.legal);
                        bufs.legal.contains(&tree.edges(current)[f].action)
                    } {
                        f
                    } else {
                        // Forced edge illegal — select among legal edges instead.
                        // bufs.legal is already populated from the check above.
                        let edges = tree.edges(current);
                        bufs.legal_edges.clear();
                        for (i, edge) in edges.iter().enumerate() {
                            if bufs.legal.contains(&edge.action) {
                                bufs.legal_edges.push(i);
                            }
                        }
                        if bufs.legal_edges.is_empty() {
                            break;
                        }
                        select_decision(tree, current, sign, &bufs.legal_edges)
                    }
                } else {
                    let edges = tree.edges(current);
                    if filter_legal {
                        bufs.legal.clear();
                        state.legal_actions(&mut bufs.legal);
                        bufs.legal_edges.clear();
                        for (i, edge) in edges.iter().enumerate() {
                            if bufs.legal.contains(&edge.action) {
                                bufs.legal_edges.push(i);
                            }
                        }
                        if bufs.legal_edges.is_empty() {
                            break;
                        }
                        select_decision(tree, current, sign, &bufs.legal_edges)
                    } else {
                        bufs.legal_edges.clear();
                        bufs.legal_edges.extend(0..edges.len());
                        select_decision(tree, current, sign, &bufs.legal_edges)
                    }
                }
            }
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
            ExpandResult::NeedsEval(sign) => {
                let state_key = state.state_key();
                return SimResult::NeedsEval {
                    state,
                    context: Phase::Simulating {
                        sign,
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

// ── Gumbel helpers ────────────────────────────────────────────────────

/// Normalize Q into [0, 1] from the current player's perspective.
///
/// Flips Q and bounds for the minimizing player so that higher normalized
/// values always mean "better for me".
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
    root_sign: f32,
    config: &Config,
    rng: &mut fastrand::Rng,
    legal_edges: Option<Vec<usize>>,
) -> GumbelState {
    let edges = tree.edges(root);

    let root_logits: Vec<f32> = edges.iter().map(|e| e.logit).collect();
    let scale = config.gumbel_scale;
    let gumbel_scores: Vec<f32> = root_logits
        .iter()
        .map(|&l| scale * sample_gumbel(rng) + l)
        .collect();

    // Build scored list: only include legal edges when filtering.
    let mut scored: Vec<(usize, f32)> = if let Some(ref le) = legal_edges {
        le.iter().map(|&i| (i, gumbel_scores[i])).collect()
    } else {
        gumbel_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect()
    };
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Gumbel-Top-k: score = g + logit, take top m
    // At least 1 candidate is needed to select an action even with 0 simulations.
    let m = (config.num_sampled_actions as usize)
        .min(scored.len())
        .min(config.num_simulations as usize)
        .max(1);
    let candidates: Vec<usize> = scored.iter().take(m).map(|&(i, _)| i).collect();

    let schedule = Schedule::new(config.num_simulations as usize, candidates.len());

    GumbelState {
        root_sign,
        gumbel_scores,
        root_logits,
        candidates,
        schedule,
        sim_index: 0,
        q_min: root_value,
        q_max: root_value,
        legal_edges,
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
    let q_norm = normalize_q(cq, gs.q_min, gs.q_max, gs.root_sign);
    let s = sigma(q_norm, ctx.max_visits, config.c_visit, config.c_scale);
    gs.gumbel_scores[edge_idx] + s
}

/// Halve candidates at end of a Sequential Halving phase.
/// Scores all candidates by g(a) + logit(a) + σ(completedQ(a)) and keeps
/// the top half (ceil).
fn halve_candidates(gs: &mut GumbelState, tree: &Tree, root: NodeId, config: &Config) {
    let ctx = RootContext::new(tree, root);

    let mut scored: Vec<(usize, f32)> = gs
        .candidates
        .iter()
        .map(|&idx| (idx, score_candidate(idx, gs, &ctx, config)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    let keep = scored.len().div_ceil(2);
    gs.candidates = scored.into_iter().take(keep).map(|(idx, _)| idx).collect();
}

/// Advance the simulation counter and check for phase boundaries.
///
/// Called at queue time for both synchronously completed simulations and
/// in-flight leaves (before eval returns).  Q bounds are updated *before*
/// virtual loss is applied (see call site), preventing permanent widening
/// from pessimistic values.
fn advance_sim(
    gs: &mut GumbelState,
    tree: &Tree,
    path: &[(NodeId, usize)],
    root: NodeId,
    config: &Config,
) {
    update_q_bounds(gs, tree, path);
    gs.sim_index += 1;
    if gs.schedule.should_halve(gs.sim_index) {
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
///
/// When `legal` is `Some`, only edges whose actions appear in the set are
/// included in the policy and considered for `selected_action`.
fn visit_count_result<G: Game>(
    tree: &Tree,
    root: NodeId,
    legal: Option<&[usize]>,
    network_value: f32,
    pv_depth: u32,
    max_depth: u32,
) -> SearchResult {
    let edges = tree.edges(root);
    let is_legal = |action: usize| legal.is_none_or(|l| l.contains(&action));

    let total_visits: u32 = edges
        .iter()
        .filter(|e| is_legal(e.action))
        .map(|e| e.visits)
        .sum();
    let mut policy = vec![0.0f32; G::NUM_ACTIONS];
    let mut best_action = edges
        .iter()
        .find(|e| is_legal(e.action))
        .map(|e| e.action)
        .unwrap_or(0);
    let mut best_visits = 0;
    if total_visits > 0 {
        for edge in edges {
            if !is_legal(edge.action) {
                continue;
            }
            policy[edge.action] = edge.visits as f32 / total_visits as f32;
            if edge.visits > best_visits {
                best_visits = edge.visits;
                best_action = edge.action;
            }
        }
    }
    let prior_top1_action = edges
        .iter()
        .filter(|e| is_legal(e.action))
        .max_by(|a, b| a.logit.total_cmp(&b.logit))
        .map(|e| e.action)
        .unwrap_or(best_action);
    let children_q: Vec<(usize, f32)> = edges
        .iter()
        .filter(|e| is_legal(e.action))
        .filter_map(|e| e.child.map(|c| (e.action, tree.q(c))))
        .collect();

    SearchResult {
        policy,
        wdl: tree.wdl(root),
        selected_action: best_action,
        network_value,
        children_q,
        prior_top1_action,
        pv_depth,
        max_depth,
    }
}

/// Extract Gumbel search result.
fn extract_gumbel_result<G: Game>(
    tree: &Tree,
    root: NodeId,
    gs: &GumbelState,
    config: &Config,
    network_value: f32,
    pv_depth: u32,
    max_depth: u32,
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

    // Improved policy (training target): softmax(logit + σ(completedQ)) over legal edges.
    // Unvisited edges use v_mix as their completedQ estimate (see completed_q).
    // When legal_edges is set (SO-ISMCTS tree reuse), illegal edges get -inf
    // so they receive zero probability after softmax.
    let mut improved_logits = Vec::with_capacity(ctx.edges.len());
    for (i, edge) in ctx.edges.iter().enumerate() {
        if gs.legal_edges.as_ref().is_some_and(|le| !le.contains(&i)) {
            improved_logits.push(f32::NEG_INFINITY);
        } else {
            let cq = completed_q(ctx.tree, edge, ctx.vmix_val);
            let q_norm = normalize_q(cq, gs.q_min, gs.q_max, gs.root_sign);
            let s = sigma(q_norm, ctx.max_visits, config.c_visit, config.c_scale);
            improved_logits.push(gs.root_logits[i] + s);
        }
    }
    softmax(&mut improved_logits);

    let mut policy = vec![0.0f32; G::NUM_ACTIONS];
    for (edge, &prob) in ctx.edges.iter().zip(&improved_logits) {
        policy[edge.action] = prob;
    }

    // Network's top-1 action (highest raw prior logit, filtered to legal edges)
    let prior_top1_action = gs
        .root_logits
        .iter()
        .enumerate()
        .filter(|(i, _)| gs.legal_edges.as_ref().is_none_or(|le| le.contains(i)))
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
        wdl: tree.wdl(root),
        selected_action,
        network_value,
        children_q,
        prior_top1_action,
        pv_depth,
        max_depth,
    }
}

/// Recursively snapshot a tree node and its children.
fn snapshot_node(
    tree: &Tree,
    node: NodeId,
    action: Option<usize>,
    parent_player: Option<u8>,
    depth_remaining: usize,
) -> TreeNodeSnapshot {
    // Who acts at this node (used as parent_player for children).
    let this_player = match tree.kind(node) {
        NodeKind::Decision(sign) => Some(if *sign > 0.0 { 0u8 } else { 1 }),
        _ => None,
    };

    let kind_str = match tree.kind(node) {
        NodeKind::Terminal => "terminal",
        NodeKind::Decision(_) => "decision",
        NodeKind::Chance => "chance",
    };

    let children = if depth_remaining == 0 {
        Vec::new()
    } else {
        tree.edges(node)
            .iter()
            .filter_map(|edge| {
                edge.child.map(|child| {
                    snapshot_node(
                        tree,
                        child,
                        Some(edge.action),
                        this_player,
                        depth_remaining - 1,
                    )
                })
            })
            .collect()
    };

    let data = &tree[node];
    TreeNodeSnapshot {
        action,
        label: None,
        wdl: data.wdl,
        visits: data.total_visits,
        kind: kind_str,
        player: parent_player,
        children,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Evaluator, RolloutEvaluator};

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
                Status::Ongoing
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
                None => Status::Ongoing,
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
                assert_eq!(result.wdl, [1.0, 0.0, 0.0]);
                assert!(result.policy.iter().all(|&p| p == 0.0));
            }
            Step::NeedsEval(_) => panic!("terminal root should not need eval"),
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

    #[test]
    fn sequential_halving_halves_candidates() {
        // With 4 candidates, after one phase we should have 2
        let evaluator = RolloutEvaluator::default();
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

    #[test]
    fn snapshot_after_search() {
        let evaluator = RolloutEvaluator::default();
        let config = Config {
            num_simulations: 100,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(TrivialGame::new(), config);
        let _result = run_to_completion(&mut search, &evaluator, &mut rng);

        let snap = search
            .snapshot()
            .expect("snapshot should exist after search");
        assert_eq!(snap.edges.len(), 2);
        assert!(snap.total_simulations > 0);
        // All edges should have improved policy
        for edge in &snap.edges {
            assert!(edge.improved_policy.is_some());
        }
        // Improved policy should sum to ~1.0
        let ip_sum: f32 = snap.edges.iter().filter_map(|e| e.improved_policy).sum();
        assert!(
            (ip_sum - 1.0).abs() < 0.01,
            "improved policy sum = {ip_sum}"
        );
    }

    #[test]
    fn snapshot_subtree_depth() {
        let evaluator = RolloutEvaluator::default();
        let config = Config {
            num_simulations: 200,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(TwoStepGame::new(), config);
        let _result = run_to_completion(&mut search, &evaluator, &mut rng);

        // Depth 0: root only, no children
        let snap0 = search.snapshot_subtree(0).unwrap();
        assert!(snap0.children.is_empty());
        assert!(snap0.visits > 0);

        // Depth 1: root + immediate children
        let snap1 = search.snapshot_subtree(1).unwrap();
        assert!(!snap1.children.is_empty());
        for child in &snap1.children {
            assert!(child.children.is_empty()); // depth limit stops here
        }

        // Depth 2: can see grandchildren (terminal nodes)
        let snap2 = search.snapshot_subtree(2).unwrap();
        let has_grandchildren = snap2.children.iter().any(|c| !c.children.is_empty());
        assert!(has_grandchildren, "depth 2 should reach terminal children");
    }

    #[test]
    fn snapshot_at_path_works() {
        let evaluator = RolloutEvaluator::default();
        let config = Config {
            num_simulations: 200,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(TwoStepGame::new(), config);
        let _result = run_to_completion(&mut search, &evaluator, &mut rng);

        // Navigate to action 0's subtree
        let snap = search.snapshot_at_path(&[0], 1).unwrap();
        assert_eq!(snap.action, Some(0));
        assert_eq!(snap.kind, "decision");

        // Invalid path returns None
        assert!(search.snapshot_at_path(&[99], 1).is_none());
    }

    #[test]
    fn snapshot_none_before_search() {
        let search = Search::new(TrivialGame::new(), Config::default());
        assert!(search.snapshot().is_none());
        assert!(search.snapshot_subtree(1).is_none());
    }

    /// Game that transitions into a degenerate state: Ongoing but no chance
    /// outcomes and no legal actions (e.g. Catan's DevCardDraw with an
    /// exhausted pool, or StealResolve with empty opponent hand).
    #[derive(Clone)]
    struct DeadEndGame {
        stuck: bool,
    }

    impl Game for DeadEndGame {
        const NUM_ACTIONS: usize = 2;

        fn status(&self) -> Status {
            // Not terminal, even when stuck.
            Status::Ongoing
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

    /// A degenerate non-terminal root (no chance outcomes, no legal actions)
    /// panics with diagnostic info so the game logic bug can be found.
    #[test]
    #[should_panic(expected = "degenerate non-terminal root")]
    fn degenerate_root_panics_with_diagnostics() {
        let evaluator = RolloutEvaluator::default();
        let config = Config {
            num_simulations: 10,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let mut search = Search::new(DeadEndGame { stuck: true }, config);
        let _result = run_to_completion(&mut search, &evaluator, &mut rng);
    }
}
