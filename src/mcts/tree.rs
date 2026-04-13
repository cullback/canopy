use std::collections::{HashMap, VecDeque};
use std::ops::{Index, IndexMut};

use crate::eval::{Evaluation, wdl_from_scalar};
use crate::game::{Game, Status};

// ── NodeId ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct NodeId(u32);

// ── Edge ─────────────────────────────────────────────────────────────

pub(super) struct Edge {
    pub action: usize,
    pub child: Option<NodeId>,
    /// Softmax probability (for v_mix weighting, chance sampling).
    pub prior: f32,
    /// Raw policy logit (for Gumbel/improved-policy selection).
    pub logit: f32,
    pub visits: u32,
    /// In-flight simulations treated as losses (leaf parallelism).
    pub virtual_losses: u32,
}

impl Edge {
    fn new_decision(action: usize, prior: f32, logit: f32) -> Self {
        Self {
            action,
            child: None,
            prior,
            logit,
            visits: 0,
            virtual_losses: 0,
        }
    }

    fn new_chance((action, weight): (usize, u32)) -> Self {
        Self {
            action,
            child: None,
            prior: weight as f32,
            logit: 0.0,
            visits: 0,
            virtual_losses: 0,
        }
    }
}

// ── Node types ───────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) enum NodeKind {
    Terminal,
    Decision(f32),
    Chance,
}

pub(super) struct NodeData {
    pub kind: NodeKind,
    pub total_visits: u32,
    /// Raw network WDL (P1 perspective), preserved for backup numerator.
    pub utility_wdl: [f32; 3],
    /// Search-averaged WDL (P1 perspective), updated by backup.
    pub wdl: [f32; 3],
}

impl NodeData {
    fn new(kind: NodeKind, wdl: [f32; 3]) -> Self {
        let total_visits = match kind {
            NodeKind::Chance => 0,
            _ => 1,
        };
        Self {
            kind,
            total_visits,
            utility_wdl: wdl,
            wdl,
        }
    }
}

// ── Arena internals ──────────────────────────────────────────────────

struct NodeEntry {
    data: NodeData,
    edge_start: u32,
    edge_count: u32,
}

// ── Scratch buffers ──────────────────────────────────────────────────

#[derive(Default)]
pub(super) struct Bufs {
    pub actions: Vec<usize>,
    pub chances: Vec<(usize, u32)>,
    pub path: Vec<(NodeId, usize)>,
    /// Reusable buffer for softmax / improved-policy computation during
    /// interior selection (avoids per-node allocation).
    pub scratch: Vec<f32>,
    /// Scratch buffer for `Game::legal_actions()` output during SO-ISMCTS filtering.
    pub legal: Vec<usize>,
    /// Edge indices that survive the legal-action intersection.
    pub legal_edges: Vec<usize>,
    /// Recycled path Vecs to avoid per-leaf allocation.
    spare_paths: Vec<Vec<(NodeId, usize)>>,
    /// Recycled actions Vecs to avoid per-leaf allocation.
    spare_actions: Vec<Vec<usize>>,
}

impl Bufs {
    /// Move path out, swapping in a spare (or empty Vec) so the next
    /// `simulate` call reuses an existing allocation.
    pub fn take_path(&mut self) -> Vec<(NodeId, usize)> {
        let spare = self.spare_paths.pop().unwrap_or_default();
        std::mem::replace(&mut self.path, spare)
    }

    /// Move actions out, swapping in a spare.
    pub fn take_actions(&mut self) -> Vec<usize> {
        let spare = self.spare_actions.pop().unwrap_or_default();
        std::mem::replace(&mut self.actions, spare)
    }

    /// Return a used path Vec to the pool.
    pub fn reclaim_path(&mut self, mut v: Vec<(NodeId, usize)>) {
        v.clear();
        self.spare_paths.push(v);
    }

    /// Return a used actions Vec to the pool.
    pub fn reclaim_actions(&mut self, mut v: Vec<usize>) {
        v.clear();
        self.spare_actions.push(v);
    }
}

// ── Expand result ────────────────────────────────────────────────────

pub(super) enum ExpandResult {
    Leaf(NodeId),
    Chance(NodeId),
    NeedsEval(f32),
}

// ── Tree ─────────────────────────────────────────────────────────────

#[derive(Default)]
pub(super) struct Tree {
    nodes: Vec<NodeEntry>,
    edges: Vec<Edge>,
    table: HashMap<u64, NodeId>,
}

impl Tree {
    // ── Node/edge access ─────────────────────────────────────────

    fn edge_range(&self, id: NodeId) -> std::ops::Range<usize> {
        let e = &self.nodes[id.0 as usize];
        e.edge_start as usize..(e.edge_start + e.edge_count) as usize
    }

    pub fn edges(&self, id: NodeId) -> &[Edge] {
        &self.edges[self.edge_range(id)]
    }

    pub fn edges_mut(&mut self, id: NodeId) -> &mut [Edge] {
        let r = self.edge_range(id);
        &mut self.edges[r]
    }

    /// Maximum visit count across all edges of a node (0 if no edges).
    pub fn max_edge_visits(&self, id: NodeId) -> u32 {
        self.edges(id).iter().map(|e| e.visits).max().unwrap_or(0)
    }

    pub fn set_child(&mut self, parent: NodeId, edge_idx: usize, child: NodeId) {
        self.edges_mut(parent)[edge_idx].child = Some(child);
    }

    pub fn q(&self, id: NodeId) -> f32 {
        let w = self.nodes[id.0 as usize].data.wdl;
        w[0] - w[2]
    }

    pub fn wdl(&self, id: NodeId) -> [f32; 3] {
        self.nodes[id.0 as usize].data.wdl
    }

    pub fn utility(&self, id: NodeId) -> f32 {
        let w = self.nodes[id.0 as usize].data.utility_wdl;
        w[0] - w[2]
    }

    pub fn kind(&self, id: NodeId) -> &NodeKind {
        &self.nodes[id.0 as usize].data.kind
    }

    #[allow(dead_code)]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ── Insert ───────────────────────────────────────────────────

    pub fn insert(
        &mut self,
        state_key: Option<u64>,
        kind: NodeKind,
        wdl: [f32; 3],
        edges: impl Iterator<Item = Edge>,
    ) -> NodeId {
        let edge_start = self.edges.len() as u32;
        self.edges.extend(edges);
        let edge_count = self.edges.len() as u32 - edge_start;
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(NodeEntry {
            data: NodeData::new(kind, wdl),
            edge_start,
            edge_count,
        });
        if let Some(key) = state_key {
            self.table.insert(key, id);
        }
        id
    }

    /// Look up a state key in the transposition table.
    pub fn lookup(&self, key: u64) -> Option<NodeId> {
        self.table.get(&key).copied()
    }

    // ── Expand ───────────────────────────────────────────────────

    pub fn try_expand<G: Game>(&mut self, state: &G, bufs: &mut Bufs) -> ExpandResult {
        let state_key = state.state_key();

        if let Some(&existing) = state_key.and_then(|k| self.table.get(&k)) {
            return ExpandResult::Leaf(existing);
        }

        match state.status() {
            Status::Terminal(reward) => {
                let id = self.insert(
                    state_key,
                    NodeKind::Terminal,
                    wdl_from_scalar(reward),
                    std::iter::empty(),
                );
                ExpandResult::Leaf(id)
            }
            Status::Ongoing => {
                state.chance_outcomes(&mut bufs.chances);
                if !bufs.chances.is_empty() {
                    let edges = bufs.chances.drain(..).map(Edge::new_chance);
                    let id = self.insert(state_key, NodeKind::Chance, [0.0, 1.0, 0.0], edges);
                    return ExpandResult::Chance(id);
                }

                bufs.actions.clear();
                state.legal_actions(&mut bufs.actions);

                // Degenerate state: no chance outcomes AND no legal actions.
                // Can arise from SO-ISMCTS determinization inconsistencies
                // (e.g. a chance node whose outcome pool was exhausted by a
                // different determinization). Treat as terminal draw.
                if bufs.actions.is_empty() {
                    let id = self.insert(
                        state_key,
                        NodeKind::Terminal,
                        [0.0, 1.0, 0.0],
                        std::iter::empty(),
                    );
                    return ExpandResult::Leaf(id);
                }

                ExpandResult::NeedsEval(state.current_sign())
            }
        }
    }

    pub fn complete_expand(
        &mut self,
        eval: &Evaluation,
        actions: &[usize],
        sign: f32,
        state_key: Option<u64>,
    ) -> NodeId {
        let priors = crate::utils::softmax_masked(&eval.policy_logits, actions);
        let edges = actions.iter().copied().zip(priors).map(|(action, prior)| {
            let logit = eval.policy_logits[action];
            Edge::new_decision(action, prior, logit)
        });
        self.insert(state_key, NodeKind::Decision(sign), eval.wdl, edges)
    }

    // ── Backprop ─────────────────────────────────────────────────

    /// Walk the simulation path, incrementing edge visits and recomputing Q.
    ///
    /// Backprop walks only the current traversal path. Transposed nodes (reachable
    /// via multiple paths) receive Q updates from whichever path visits them —
    /// their Q is read correctly by all parents on the *next* visit, so the
    /// approximation is self-correcting. Full DAG backprop (upward BFS from every
    /// modified node) would be more precise but adds parent pointers, cycle
    /// detection, and O(tree) worst-case cost per simulation. Standard practice
    /// (AlphaZero, KataGo) is path-only backprop even with deduplication.
    pub fn backprop(&mut self, path: &[(NodeId, usize)]) {
        for &(nid, eidx) in path {
            self.edges_mut(nid)[eidx].visits += 1;
        }
        self.recompute_q(path);
    }

    /// Recompute Q values along a path (leaf-to-root), accounting for virtual losses.
    ///
    /// For each node on the path, re-derives Q from its edges' visits and children.
    /// Virtual-loss visits contribute a pessimistic value (`-sign` for
    /// decision nodes) instead of the child's Q, discouraging re-selection of
    /// in-flight edges. When `virtual_losses == 0` everywhere, this degenerates
    /// to the standard formula.
    pub fn recompute_q(&mut self, path: &[(NodeId, usize)]) {
        for &(nid, _) in path.iter().rev() {
            // Virtual loss WDL (P1 perspective):
            // Decision(+1): [0, 0, 1] — pessimistic for the acting player P1
            // Decision(-1): [1, 0, 0] — pessimistic for P2 (P1 winning is bad for P2)
            // Chance: [0, 1, 0] (neutral draw)
            let (is_chance, vloss_wdl) = match self[nid].kind {
                NodeKind::Decision(sign) => {
                    if sign > 0.0 {
                        (false, [0.0, 0.0, 1.0])
                    } else {
                        (false, [1.0, 0.0, 0.0])
                    }
                }
                _ => (matches!(self[nid].kind, NodeKind::Chance), [0.0, 1.0, 0.0]),
            };

            let (sum_edge_visits, wwdl) = {
                let edges = self.edges(nid);
                let sum = edges.iter().map(|e| e.visits).sum::<u32>();
                let mut wwdl = [0.0f32; 3];
                for edge in edges {
                    let real_visits = edge.visits - edge.virtual_losses;
                    if let Some(child_id) = edge.child {
                        let child_wdl = self[child_id].wdl;
                        for k in 0..3 {
                            wwdl[k] += real_visits as f32 * child_wdl[k];
                        }
                    }
                    for k in 0..3 {
                        wwdl[k] += edge.virtual_losses as f32 * vloss_wdl[k];
                    }
                }
                (sum, wwdl)
            };

            let node = &mut self[nid];
            if is_chance {
                node.total_visits = sum_edge_visits;
                if sum_edge_visits > 0 {
                    let denom = sum_edge_visits as f32;
                    for (dst, &src) in node.wdl.iter_mut().zip(&wwdl) {
                        *dst = src / denom;
                    }
                } else {
                    node.wdl = [0.0, 1.0, 0.0];
                }
            } else {
                node.total_visits = 1 + sum_edge_visits;
                let denom = node.total_visits as f32;
                let util = node.utility_wdl;
                for (k, dst) in node.wdl.iter_mut().enumerate() {
                    *dst = (util[k] + wwdl[k]) / denom;
                }
            }
        }
    }

    /// Mark edges along `path` as having an in-flight simulation (virtual loss).
    ///
    /// Increments both `visits` and `virtual_losses` on each edge, then
    /// recomputes Q. The extra visits with pessimistic Q discourage other
    /// simulations from selecting the same path.
    pub fn apply_virtual_loss(&mut self, path: &[(NodeId, usize)]) {
        for &(nid, eidx) in path {
            let edge = &mut self.edges_mut(nid)[eidx];
            edge.visits += 1;
            edge.virtual_losses += 1;
        }
        self.recompute_q(path);
    }

    /// Remove virtual loss from edges along `path` (visits stay — they become "real").
    ///
    /// Called in [`Search::supply`] before expanding the child node.
    /// Does **not** recompute Q — the caller should do so after setting the child.
    pub fn remove_virtual_loss(&mut self, path: &[(NodeId, usize)]) {
        for &(nid, eidx) in path {
            self.edges_mut(nid)[eidx].virtual_losses -= 1;
        }
    }

    // ── Compact ──────────────────────────────────────────────────

    /// Compact the tree to only nodes reachable from `root`.
    /// Remaps the transposition table. Returns the new root (always `NodeId(0)`).
    pub fn compact(&mut self, root: NodeId) -> NodeId {
        let old_len = self.nodes.len();
        let mut old_to_new: Vec<Option<NodeId>> = vec![None; old_len];

        // BFS to discover reachable nodes and assign new IDs
        let mut queue = VecDeque::new();
        old_to_new[root.0 as usize] = Some(NodeId(0));
        queue.push_back(root);
        let mut next_id = 1u32;

        while let Some(nid) = queue.pop_front() {
            let entry = &self.nodes[nid.0 as usize];
            let start = entry.edge_start as usize;
            let end = start + entry.edge_count as usize;
            for edge in &self.edges[start..end] {
                if let Some(child) = edge.child
                    && old_to_new[child.0 as usize].is_none()
                {
                    old_to_new[child.0 as usize] = Some(NodeId(next_id));
                    next_id += 1;
                    queue.push_back(child);
                }
            }
        }

        // Rust doesn't allow moving out of a Vec by index (it would leave a hole),
        // so we wrap each element in Option and use .take() to move ownership out.
        // This costs one extra allocation + bool per element; an alternative would
        // be unsafe swap-remove or sentinel values, but this is only called once
        // per action so clarity wins over micro-optimization.
        let mut old_nodes: Vec<Option<NodeEntry>> = std::mem::take(&mut self.nodes)
            .into_iter()
            .map(Some)
            .collect();
        let mut old_edges: Vec<Option<Edge>> = std::mem::take(&mut self.edges)
            .into_iter()
            .map(Some)
            .collect();

        // Build new-to-old mapping for iteration in new-ID order
        let mut new_to_old: Vec<u32> = vec![0; next_id as usize];
        for (old_idx, mapped) in old_to_new.iter().enumerate() {
            if let Some(new_id) = mapped {
                new_to_old[new_id.0 as usize] = old_idx as u32;
            }
        }

        self.nodes = Vec::with_capacity(next_id as usize);
        self.edges = Vec::new();

        for &old_id in new_to_old.iter().take(next_id as usize) {
            let old_idx = old_id as usize;
            let old_entry = old_nodes[old_idx].take().unwrap();

            let edge_start = self.edges.len() as u32;
            let old_start = old_entry.edge_start as usize;
            let old_end = old_start + old_entry.edge_count as usize;
            for slot in &mut old_edges[old_start..old_end] {
                let mut edge = slot.take().unwrap();
                edge.child = edge.child.and_then(|c| old_to_new[c.0 as usize]);
                self.edges.push(edge);
            }
            let edge_count = self.edges.len() as u32 - edge_start;

            self.nodes.push(NodeEntry {
                data: old_entry.data,
                edge_start,
                edge_count,
            });
        }

        // Remap transposition table
        self.table.retain(|_, old_id| {
            if let Some(new_id) = old_to_new[old_id.0 as usize] {
                *old_id = new_id;
                true
            } else {
                false
            }
        });

        NodeId(0)
    }

    // ── Clear ─────────────────────────────────────────────────────

    /// Remove all nodes and edges but keep the underlying `Vec` capacities
    /// so the next search can reuse the allocations.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.table.clear();
    }

    // ── Navigation ───────────────────────────────────────────────

    /// Find the child reached by `action` from `node`, if any.
    pub fn child_for_action(&self, node: NodeId, action: usize) -> Option<NodeId> {
        self.edges(node)
            .iter()
            .find(|e| e.action == action)
            .and_then(|e| e.child)
    }

    /// Return the child of the most-visited edge (used to walk through chance
    /// nodes when the actual outcome wasn't explored).
    pub fn best_chance_child(&self, node: NodeId) -> Option<NodeId> {
        self.edges(node)
            .iter()
            .filter_map(|e| e.child.map(|c| (c, e.visits)))
            .max_by_key(|&(_, v)| v)
            .map(|(c, _)| c)
    }

    /// Sample a chance edge proportional to weights (stored as f32 in `prior`).
    pub fn sample_chance_edge(&self, node: NodeId, rng: &mut fastrand::Rng) -> usize {
        let edges = self.edges(node);
        let total: u32 = edges.iter().map(|e| e.prior as u32).sum();
        let mut r = rng.u32(0..total);
        for (i, edge) in edges.iter().enumerate() {
            let w = edge.prior as u32;
            if r < w {
                return i;
            }
            r -= w;
        }
        edges.len() - 1
    }
}

impl Index<NodeId> for Tree {
    type Output = NodeData;

    fn index(&self, id: NodeId) -> &NodeData {
        &self.nodes[id.0 as usize].data
    }
}

impl IndexMut<NodeId> for Tree {
    fn index_mut(&mut self, id: NodeId) -> &mut NodeData {
        &mut self.nodes[id.0 as usize].data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_retains_reachable() {
        let mut tree = Tree::default();

        // A(0) -> B(1) -> C(2)
        let a = tree.insert(
            None,
            NodeKind::Terminal,
            [0.0, 1.0, 0.0],
            std::iter::once(Edge::new_chance((0, 1))),
        );
        let b = tree.insert(
            None,
            NodeKind::Terminal,
            [0.0, 1.0, 0.0],
            std::iter::once(Edge::new_chance((0, 1))),
        );
        let c = tree.insert(
            None,
            NodeKind::Terminal,
            [0.0, 1.0, 0.0],
            std::iter::empty(),
        );

        // Patch child pointers
        tree.set_child(a, 0, b);
        tree.set_child(b, 0, c);

        assert_eq!(tree.node_count(), 3);

        // Reroot at B — only B and C should survive
        let new_root = tree.compact(b);
        assert_eq!(new_root, NodeId(0));
        assert_eq!(tree.node_count(), 2);

        let edges = tree.edges(NodeId(0));
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].child, Some(NodeId(1)));
        assert_eq!(tree.edges(NodeId(1)).len(), 0);
    }
}
