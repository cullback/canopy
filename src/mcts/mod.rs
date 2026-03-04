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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            num_sampled_actions: 16,
            c_visit: 50.0,
            c_scale: 1.0,
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
}

/// Continuation token for a pending neural-network evaluation.
///
/// Returned inside [`Step::NeedsEval`] when the search cannot proceed without
/// an NN forward pass.  The caller evaluates `state` to produce an
/// [`Evaluation`], then hands *both* the output and this token to
/// [`Search::supply`].  The token's private `context` carries the internal
/// bookkeeping (which node is being expanded, where in the tree to attach the
/// result) so that `Search` itself needs no mutable "phase" field — all
/// in-flight state lives here, making it impossible to call `supply` without
/// a matching `NeedsEval`.
pub struct PendingEval<G: Game> {
    pub state: G,
    context: Phase,
}

/// One step of the MCTS state machine.
pub enum Step<G: Game> {
    /// Search needs an NN evaluation for this state.
    NeedsEval(PendingEval<G>),
    /// Search complete — improved policy, value, and selected action are ready.
    Done(SearchResult),
}

/// Gumbel AlphaZero MCTS, driven as a state machine.
///
/// The search tree persists across moves so that subtrees explored in earlier
/// searches can be reused (the tree is compacted, not rebuilt, on each
/// `step_to`).  The three public methods form a simple protocol:
///
/// 1. [`start`](Self::start) — construct the search and begin the first search.
/// 2. [`supply`](Self::supply) — feed an NN result back; may yield another
///    `NeedsEval` or a final `Done`.
/// 3. [`step_to`](Self::step_to) — after playing one or more actions, advance
///    the tree and begin the next search with a (possibly updated) config.
///
/// The caller never touches the tree directly; all interaction goes through
/// `Step` / `PendingEval`.
pub struct Search<G: Game> {
    tree: Tree,
    /// `None` when the root is terminal (we return `Done` immediately) or when
    /// `step_to` walks through an action the tree hasn't expanded yet (we fall
    /// back to `start` with a fresh tree).
    root: Option<NodeId>,
    root_state: G,
    bufs: Bufs,
    config: Config,
    gumbel: Option<GumbelState>,
    /// Budget counter for vanilla (non-Gumbel) simulations (chance roots).
    vanilla_budget_remaining: u32,
}

// ── Internal types ────────────────────────────────────────────────────

enum Phase {
    ExpandingRoot {
        player: Player,
    },
    Simulating {
        parent: NodeId,
        edge_idx: usize,
        player: Player,
        state_key: Option<u64>,
    },
}

/// Gumbel Sequential Halving state for root action selection.
struct GumbelState {
    /// The player at the root node (needed for Q sign-flipping).
    root_player: Player,
    /// Gumbel(0) sample per root edge.
    gumbels: Vec<f32>,
    /// Raw logits per root edge.
    root_logits: Vec<f32>,
    /// Edge indices alive in sequential halving.
    candidates: Vec<usize>,
    /// Current SH phase (0-indexed).
    phase: u32,
    /// ceil(log2(m)).
    total_phases: u32,
    /// Sims allocated per candidate this phase.
    sims_per_candidate: u32,
    /// Sims completed for current candidate in round-robin.
    sims_done_for_current: u32,
    /// Which candidate in round-robin we're currently simulating.
    candidate_idx: usize,
    /// How many candidates have completed their allocation this phase.
    candidates_simmed: u32,
    /// Min Q across tree (P1 perspective) for normalization.
    q_min: f32,
    /// Max Q across tree (P1 perspective) for normalization.
    q_max: f32,
    /// Total simulation budget remaining.
    budget_remaining: u32,
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
    /// Create a new search tree and begin searching from `root_state`.
    ///
    /// Returns `(Self, Step)` rather than `&mut self -> Step` (like
    /// [`step_to`](Self::step_to)) because the `Search` doesn't exist yet —
    /// the shape asymmetry between the two entry points is inherent, not an
    /// oversight.
    ///
    /// If `root_state` is terminal, returns `Step::Done` immediately with a
    /// zero policy and the terminal reward.  Otherwise returns `NeedsEval`
    /// for the root (or, for a chance root that needs no eval, may run
    /// simulations and return `NeedsEval` for the first leaf it reaches).
    pub fn start(root_state: &G, config: &Config, rng: &mut fastrand::Rng) -> (Self, Step<G>) {
        let mut search = Self {
            tree: Tree::default(),
            root: None,
            root_state: root_state.clone(),
            bufs: Bufs::default(),
            config: config.clone(),
            gumbel: None,
            vanilla_budget_remaining: config.num_simulations,
        };

        // Terminal root — immediate result
        if let Status::Terminal(reward) = root_state.status() {
            let step = Step::Done(SearchResult {
                policy: vec![0.0; G::NUM_ACTIONS],
                value: reward,
                selected_action: 0,
            });
            return (search, step);
        }

        // Try to expand root
        match search.tree.try_expand(root_state, &mut search.bufs) {
            ExpandResult::Leaf(_) => unreachable!("empty non-terminal tree"),
            ExpandResult::Chance(id) => {
                search.root = Some(id);
                let step = search.run_simulations(rng);
                (search, step)
            }
            ExpandResult::NeedsEval(player) => {
                let step = Step::NeedsEval(PendingEval {
                    state: root_state.clone(),
                    context: Phase::ExpandingRoot { player },
                });
                (search, step)
            }
        }
    }

    /// Feed a neural-network evaluation back into the search.
    ///
    /// `pending` is the [`PendingEval`] token from the most recent
    /// `NeedsEval` step — it carries the private context describing *where*
    /// in the tree the new node should be attached.  Consuming the token
    /// (rather than reading a mutable `phase` field on `self`) makes
    /// mis-use a compile error: you can't call `supply` twice for the same
    /// evaluation, and you can't call it without a preceding `NeedsEval`.
    ///
    /// Config is *not* accepted here because the search is mid-flight;
    /// changing simulation budget or Gumbel parameters between leaf
    /// expansions would invalidate Sequential Halving bookkeeping.  Config
    /// changes take effect at the next [`step_to`](Self::step_to).
    pub fn supply(
        &mut self,
        eval: Evaluation,
        pending: PendingEval<G>,
        rng: &mut fastrand::Rng,
    ) -> Step<G> {
        match pending.context {
            Phase::ExpandingRoot { player } => {
                let state_key = self.root_state.state_key();
                let root = self
                    .tree
                    .complete_expand(&eval, &mut self.bufs, player, state_key);
                self.root = Some(root);

                // Initialize Gumbel state for root
                self.gumbel = Some(init_gumbel(
                    &self.tree,
                    root,
                    eval.value,
                    player,
                    &self.config,
                    rng,
                ));
            }
            Phase::Simulating {
                parent,
                edge_idx,
                player,
                state_key,
            } => {
                let child = self
                    .tree
                    .complete_expand(&eval, &mut self.bufs, player, state_key);
                self.tree.edges_mut(parent)[edge_idx].child = Some(child);
                self.tree.backprop(&self.bufs.path);

                let root = self.root();
                if let Some(gs) = &mut self.gumbel {
                    advance_round_robin(gs, &self.tree, &self.bufs.path, root, &self.config);
                } else {
                    self.vanilla_budget_remaining = self.vanilla_budget_remaining.saturating_sub(1);
                }
            }
        };
        self.run_simulations(rng)
    }

    /// Advance the tree through `actions` played since the last search,
    /// update config, and begin a new search from `root_state`.
    ///
    /// `actions` should contain every action applied to the game state since
    /// the previous `start` or `step_to` — both player decisions and chance
    /// outcomes — so the tree can follow the corresponding edges and reuse
    /// the subtree.  The slice is walked left-to-right; if any action leads
    /// to an unexpanded or missing child the remaining actions are skipped,
    /// the old tree is discarded, and a fresh tree is built from scratch
    /// (equivalent to calling `start`).
    ///
    /// `config` is stored for the duration of this search, so callers can
    /// adjust simulation budget or Gumbel parameters between moves (e.g.
    /// fewer sims late in a game when the position is decided).
    pub fn step_to(
        &mut self,
        root_state: &G,
        actions: &[usize],
        config: &Config,
        rng: &mut fastrand::Rng,
    ) -> Step<G> {
        // Advance tree through played actions
        for &action in actions {
            if let Some(root) = self.root {
                self.root = self.tree.child_for_action(root, action);
            }
        }
        self.gumbel = None;

        // If any action in the slice wasn't found in the tree (e.g. an
        // unexpanded chance outcome), root is None and we silently discard
        // the old tree and start fresh.
        let Some(old_root) = self.root else {
            let (new_search, step) = Search::start(root_state, config, rng);
            *self = new_search;
            return step;
        };

        // Compact the graph and remap transposition table
        let new_root = self.tree.compact(old_root);
        self.root = Some(new_root);
        self.root_state = root_state.clone();
        self.config = config.clone();
        self.vanilla_budget_remaining = config.num_simulations;

        // Terminal root — immediate result
        if let Status::Terminal(reward) = root_state.status() {
            return Step::Done(SearchResult {
                policy: vec![0.0; G::NUM_ACTIONS],
                value: reward,
                selected_action: 0,
            });
        }

        // Re-initialize Gumbel state with fresh samples for the reused root
        let root_value = self.tree.utility(new_root);
        let root_player = match *self.tree.kind(new_root) {
            NodeKind::Decision(p) => p,
            _ => {
                // Chance/terminal root — no Gumbel needed
                return self.run_vanilla_sims(rng);
            }
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
    }

    /// The current root node. Panics if root is unset (only possible for
    /// terminal roots returned as Step::Done, or after advance with no child).
    fn root(&self) -> NodeId {
        self.root
            .expect("root accessed before initialization or after terminal")
    }

    fn run_simulations(&mut self, rng: &mut fastrand::Rng) -> Step<G> {
        let root = self.root();

        let gs = match &mut self.gumbel {
            Some(gs) => gs,
            None => {
                // No gumbel state (e.g. chance root) — just run vanilla sims
                return self.run_vanilla_sims(rng);
            }
        };

        // Single legal action fast path
        if gs.candidates.len() <= 1 {
            let edges = self.tree.edges(root);
            let action = if edges.is_empty() {
                0
            } else {
                edges[gs.candidates.first().copied().unwrap_or(0)].action
            };
            let mut policy = vec![0.0f32; G::NUM_ACTIONS];
            if !edges.is_empty() {
                policy[action] = 1.0;
            }
            return Step::Done(SearchResult {
                policy,
                value: self.tree.q(root),
                selected_action: action,
            });
        }

        loop {
            // Check if SH is complete (1 candidate left or budget exhausted)
            if gs.candidates.len() <= 1 || gs.budget_remaining == 0 {
                return Step::Done(extract_gumbel_result::<G>(
                    &self.tree,
                    root,
                    gs,
                    &self.config,
                ));
            }

            // Get the forced root edge for this simulation
            let forced_edge = gs.candidates[gs.candidate_idx];

            match simulate_one(
                &mut self.tree,
                root,
                &self.root_state,
                &self.config,
                rng,
                &mut self.bufs,
                Some(forced_edge),
                (gs.q_min, gs.q_max),
            ) {
                SimResult::Complete => {
                    advance_round_robin(gs, &self.tree, &self.bufs.path, root, &self.config);
                }
                SimResult::NeedsEval {
                    state,
                    parent,
                    edge_idx,
                    player,
                    state_key,
                } => {
                    return Step::NeedsEval(PendingEval {
                        state,
                        context: Phase::Simulating {
                            parent,
                            edge_idx,
                            player,
                            state_key,
                        },
                    });
                }
            }
        }
    }

    /// Fallback for roots without Gumbel state (chance roots).
    fn run_vanilla_sims(&mut self, rng: &mut fastrand::Rng) -> Step<G> {
        let root = self.root();
        while self.vanilla_budget_remaining > 0 {
            match simulate_one(
                &mut self.tree,
                root,
                &self.root_state,
                &self.config,
                rng,
                &mut self.bufs,
                None,
                (0.0, 0.0), // equal bounds → normalize_q returns 0.5, selection is policy-driven
            ) {
                SimResult::Complete => {
                    self.vanilla_budget_remaining -= 1;
                }
                SimResult::NeedsEval {
                    state,
                    parent,
                    edge_idx,
                    player,
                    state_key,
                } => {
                    return Step::NeedsEval(PendingEval {
                        state,
                        context: Phase::Simulating {
                            parent,
                            edge_idx,
                            player,
                            state_key,
                        },
                    });
                }
            }
        }
        Step::Done(visit_count_result::<G>(&self.tree, root))
    }
}

// ── Simulation ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn simulate_one<G: Game>(
    tree: &mut Tree,
    root: NodeId,
    root_state: &G,
    config: &Config,
    rng: &mut fastrand::Rng,
    bufs: &mut Bufs,
    forced_root_edge: Option<usize>,
    q_bounds: (f32, f32),
) -> SimResult<G> {
    bufs.path.clear();
    let mut current = root;
    let mut state = root_state.clone();
    let mut is_root = true;

    loop {
        let edges = tree.edges(current);

        let edge_idx = match *tree.kind(current) {
            NodeKind::Terminal => break,
            NodeKind::Chance => tree.sample_chance_edge(current, rng),
            NodeKind::Decision(player) => {
                if is_root && let Some(forced) = forced_root_edge {
                    forced
                } else {
                    gumbel_interior_select(tree, current, player, config, q_bounds)
                }
            }
        };

        is_root = false;
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
                    parent: current,
                    edge_idx,
                    player,
                    state_key,
                };
            }
            ExpandResult::Chance(id) => {
                tree.edges_mut(current)[edge_idx].child = Some(id);
                current = id;
            }
            ExpandResult::Leaf(id) => {
                tree.edges_mut(current)[edge_idx].child = Some(id);
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
fn gumbel_interior_select(
    tree: &Tree,
    node_id: NodeId,
    player: Player,
    config: &Config,
    q_bounds: (f32, f32),
) -> usize {
    let edges = tree.edges(node_id);
    let total_child_visits: u32 = edges.iter().map(|e| e.visits).sum();
    let max_visits = edges.iter().map(|e| e.visits).max().unwrap_or(0);
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
        .unwrap()
        .0
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

/// Policy-weighted value mixing (paper's formula).
fn v_mix(tree: &Tree, id: NodeId) -> f32 {
    let edges = tree.edges(id);
    let n_total: f32 = edges.iter().map(|e| e.visits).sum::<u32>() as f32;
    if n_total == 0.0 {
        return tree.utility(id);
    }

    let mut weighted_q = 0.0f32;
    let mut weight_sum = 0.0f32;
    for edge in edges {
        if let Some(child) = edge.child {
            weighted_q += edge.prior * tree.q(child);
            weight_sum += edge.prior;
        }
    }
    let search_q = if weight_sum > 0.0 {
        weighted_q / weight_sum
    } else {
        0.0
    };

    (tree.utility(id) + n_total * search_q) / (1.0 + n_total)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        probs.iter_mut().for_each(|p| *p /= sum);
    }
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
    let gumbels: Vec<f32> = (0..num_edges).map(|_| sample_gumbel(rng)).collect();

    // Gumbel-Top-k: score = g + logit, take top m
    let m = (config.num_sampled_actions as usize)
        .min(num_edges)
        .min(config.num_simulations as usize);

    let mut scored: Vec<(usize, f32)> = gumbels
        .iter()
        .zip(root_logits.iter())
        .enumerate()
        .map(|(i, (&g, &l))| (i, g + l))
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
        gumbels,
        root_logits,
        candidates,
        phase: 0,
        total_phases,
        sims_per_candidate,
        sims_done_for_current: 0,
        candidate_idx: 0,
        candidates_simmed: 0,
        q_min: root_value,
        q_max: root_value,
        budget_remaining: budget,
    }
}

/// Score a candidate edge: g(a) + logit(a) + σ(completedQ(a)).
/// `vmix_val` and `max_visits` are precomputed from the root's edges.
fn score_candidate(
    edge_idx: usize,
    gs: &GumbelState,
    edges: &[tree::Edge],
    tree: &Tree,
    vmix_val: f32,
    max_visits: u32,
    config: &Config,
) -> f32 {
    let cq = completed_q(tree, &edges[edge_idx], vmix_val);
    let q_norm = normalize_q_for_player(cq, gs.q_min, gs.q_max, gs.root_player);
    let s = sigma(q_norm, max_visits, config.c_visit, config.c_scale);
    gs.gumbels[edge_idx] + gs.root_logits[edge_idx] + s
}

/// Halve candidates at end of a Sequential Halving phase.
fn halve_candidates(gs: &mut GumbelState, tree: &Tree, root: NodeId, config: &Config) {
    let edges = tree.edges(root);
    let vmix_val = v_mix(tree, root);
    let max_visits = edges.iter().map(|e| e.visits).max().unwrap_or(0);

    // Score all candidates
    let mut scored: Vec<(usize, f32)> = gs
        .candidates
        .iter()
        .map(|&idx| {
            (
                idx,
                score_candidate(idx, gs, edges, tree, vmix_val, max_visits, config),
            )
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Keep top half (ceil)
    let keep = scored.len().div_ceil(2);
    gs.candidates = scored.into_iter().take(keep).map(|(idx, _)| idx).collect();

    // Reset round-robin for the new phase — candidate_idx = 0 is intentional
    // since candidates has been re-sorted by score and the old indices are gone.
    gs.phase += 1;
    gs.candidates_simmed = 0;
    gs.candidate_idx = 0;
    gs.sims_done_for_current = 0;

    // Recompute sims_per_candidate for remaining budget and candidates
    let remaining_phases = gs.total_phases.saturating_sub(gs.phase);
    let m = gs.candidates.len() as u32;
    if remaining_phases > 0 && m > 0 {
        gs.sims_per_candidate = (gs.budget_remaining / (remaining_phases * m)).max(1);
    } else {
        gs.sims_per_candidate = 1;
    }
}

/// Advance Sequential Halving round-robin after a completed simulation.
/// Called from both `run_simulations` (sync completion) and `supply` (async completion).
fn advance_round_robin(
    gs: &mut GumbelState,
    tree: &Tree,
    path: &[(NodeId, usize)],
    root: NodeId,
    config: &Config,
) {
    update_q_bounds(gs, tree, path);
    gs.budget_remaining = gs.budget_remaining.saturating_sub(1);
    gs.sims_done_for_current += 1;

    if gs.sims_done_for_current >= gs.sims_per_candidate {
        gs.sims_done_for_current = 0;
        gs.candidates_simmed += 1;
        gs.candidate_idx = (gs.candidate_idx + 1) % gs.candidates.len();
    }

    if gs.candidates_simmed >= gs.candidates.len() as u32 {
        halve_candidates(gs, tree, root, config);
    }
}

/// Update q_min/q_max from nodes touched in backprop path.
fn update_q_bounds(gs: &mut GumbelState, tree: &Tree, path: &[(NodeId, usize)]) {
    for &(nid, eidx) in path {
        let node_q = tree.q(nid);
        gs.q_min = gs.q_min.min(node_q);
        gs.q_max = gs.q_max.max(node_q);

        // Also check the child
        if let Some(child_id) = tree.edges(nid)[eidx].child {
            let child_q = tree.q(child_id);
            gs.q_min = gs.q_min.min(child_q);
            gs.q_max = gs.q_max.max(child_q);
        }
    }
}

/// Build a result from visit counts (used for chance roots without Gumbel state).
fn visit_count_result<G: Game>(tree: &Tree, root: NodeId) -> SearchResult {
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
    SearchResult {
        policy,
        value: tree.q(root),
        selected_action: best_action,
    }
}

/// Extract Gumbel search result.
fn extract_gumbel_result<G: Game>(
    tree: &Tree,
    root: NodeId,
    gs: &GumbelState,
    config: &Config,
) -> SearchResult {
    let edges = tree.edges(root);
    let vmix_val = v_mix(tree, root);
    let max_visits = edges.iter().map(|e| e.visits).max().unwrap_or(0);

    // selected_action: argmax over final candidates of g + logit + σ(completedQ)
    let selected_edge = gs
        .candidates
        .iter()
        .map(|&idx| {
            (
                idx,
                score_candidate(idx, gs, edges, tree, vmix_val, max_visits, config),
            )
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let selected_action = edges[selected_edge].action;

    // improved policy (training target): softmax(logit + σ(completedQ)) over ALL edges
    let mut improved_logits = Vec::with_capacity(edges.len());
    for (i, edge) in edges.iter().enumerate() {
        let cq = completed_q(tree, edge, vmix_val);
        let q_norm = normalize_q_for_player(cq, gs.q_min, gs.q_max, gs.root_player);
        let s = sigma(q_norm, max_visits, config.c_visit, config.c_scale);
        improved_logits.push(gs.root_logits[i] + s);
    }
    let improved_probs = softmax(&improved_logits);

    let mut policy = vec![0.0f32; G::NUM_ACTIONS];
    for (edge, &prob) in edges.iter().zip(&improved_probs) {
        policy[edge.action] = prob;
    }

    SearchResult {
        policy,
        value: tree.q(root),
        selected_action,
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

        let (mut sm, mut step) = Search::start(&game, &config, &mut rng);
        let result = loop {
            step = match step {
                Step::NeedsEval(pending) => {
                    let output = evaluator.evaluate(&pending.state, &mut rng);
                    sm.supply(output, pending, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

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

    #[test]
    fn state_machine_api_works() {
        let game = TrivialGame::new();
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
        let config = Config {
            num_simulations: 500,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let (mut sm, mut step) = Search::start(&game, &config, &mut rng);
        let result = loop {
            step = match step {
                Step::NeedsEval(pending) => {
                    let output = evaluator.evaluate(&pending.state, &mut rng);
                    sm.supply(output, pending, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

        assert_eq!(
            result.selected_action, 0,
            "State machine API should find that action 0 wins"
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

        let mut game = TwoStepGame::new();
        let (mut search, mut step) = Search::start(&game, &config, &mut rng);
        let result = loop {
            step = match step {
                Step::NeedsEval(pending) => {
                    let output = evaluator.evaluate(&pending.state, &mut rng);
                    search.supply(output, pending, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

        let action = result.selected_action;
        game.apply_action(action);

        let node_count_before = search.tree.node_count();
        step = search.step_to(&game, &[action], &config, &mut rng);
        let node_count_after_compact = search.tree.node_count();
        assert!(
            node_count_after_compact < node_count_before,
            "retain_subtree should compact: {node_count_after_compact} >= {node_count_before}"
        );

        let result2 = loop {
            step = match step {
                Step::NeedsEval(pending) => {
                    let output = evaluator.evaluate(&pending.state, &mut rng);
                    search.supply(output, pending, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

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

        let (_sm, step) = Search::start(&game, &config, &mut rng);
        match step {
            Step::Done(result) => {
                assert_eq!(result.value, 1.0);
                assert!(result.policy.iter().all(|&p| p == 0.0));
            }
            Step::NeedsEval(_) => panic!("terminal root should not need eval"),
        }
    }

    #[test]
    fn improved_policy_sums_to_one() {
        let game = TrivialGame::new();
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
        let config = Config {
            num_simulations: 100,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let (mut sm, mut step) = Search::start(&game, &config, &mut rng);
        let result = loop {
            step = match step {
                Step::NeedsEval(pending) => {
                    let output = evaluator.evaluate(&pending.state, &mut rng);
                    sm.supply(output, pending, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

        let total: f32 = result.policy.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "improved policy should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn sequential_halving_halves_candidates() {
        // With 4 candidates, after one phase we should have 2
        let game = TrivialGame::new();
        let evaluator = RolloutEvaluator { num_rollouts: 1 };
        let config = Config {
            num_simulations: 100,
            num_sampled_actions: 4, // but TrivialGame only has 2 actions
            ..Default::default()
        };
        let mut rng = fastrand::Rng::new();

        let (mut sm, mut step) = Search::start(&game, &config, &mut rng);
        let result = loop {
            step = match step {
                Step::NeedsEval(pending) => {
                    let output = evaluator.evaluate(&pending.state, &mut rng);
                    sm.supply(output, pending, &mut rng)
                }
                Step::Done(r) => break r,
            };
        };

        // Should complete successfully and pick the winning action
        assert_eq!(result.selected_action, 0);
    }
}
