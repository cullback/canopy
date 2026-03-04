use std::collections::{HashMap, VecDeque};
use std::ops::{Index, IndexMut};

use crate::eval::NnOutput;
use crate::game::{Game, Status};
use crate::player::Player;

// ── NodeId ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct NodeId(u32);

// ── Edge ─────────────────────────────────────────────────────────────

pub(super) struct Edge {
    pub action: usize,
    pub child: Option<NodeId>,
    /// Softmax probability (for v_mix weighting, chance sampling).
    pub prior: f32,
    /// Raw NN logit (for Gumbel/improved-policy selection).
    pub logit: f32,
    pub visits: u32,
}

impl Edge {
    fn new_decision(action: usize, prior: f32, logit: f32) -> Self {
        Self {
            action,
            child: None,
            prior,
            logit,
            visits: 0,
        }
    }

    fn new_chance((action, prior): (usize, f32)) -> Self {
        Self {
            action,
            child: None,
            prior,
            logit: 0.0,
            visits: 0,
        }
    }
}

// ── Node types ───────────────────────────────────────────────────────

pub(super) enum NodeKind {
    Terminal,
    Decision(Player),
    Chance,
}

pub(super) struct NodeData {
    pub kind: NodeKind,
    pub total_visits: u32,
    pub utility: f32,
    pub q: f32,
}

impl NodeData {
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
    pub chances: Vec<(usize, f32)>,
    pub path: Vec<(NodeId, usize)>,
}

// ── Expand result ────────────────────────────────────────────────────

pub(super) enum ExpandResult {
    Leaf(NodeId),
    Chance(NodeId),
    NeedsEval(Player),
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

    pub fn edges(&self, id: NodeId) -> &[Edge] {
        let entry = &self.nodes[id.0 as usize];
        &self.edges[entry.edge_start as usize..(entry.edge_start + entry.edge_count) as usize]
    }

    pub fn edges_mut(&mut self, id: NodeId) -> &mut [Edge] {
        let entry = &self.nodes[id.0 as usize];
        &mut self.edges[entry.edge_start as usize..(entry.edge_start + entry.edge_count) as usize]
    }

    pub fn q(&self, id: NodeId) -> f32 {
        self.nodes[id.0 as usize].data.q
    }

    pub fn utility(&self, id: NodeId) -> f32 {
        self.nodes[id.0 as usize].data.utility
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
        value: f32,
        edges: impl Iterator<Item = Edge>,
    ) -> NodeId {
        let edge_start = self.edges.len() as u32;
        self.edges.extend(edges);
        let edge_count = self.edges.len() as u32 - edge_start;
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(NodeEntry {
            data: NodeData::new(kind, value),
            edge_start,
            edge_count,
        });
        if let Some(key) = state_key {
            self.table.insert(key, id);
        }
        id
    }

    // ── Expand ───────────────────────────────────────────────────

    pub fn try_expand<G: Game>(&mut self, state: &G, bufs: &mut Bufs) -> ExpandResult {
        let state_key = state.state_key();

        if let Some(&existing) = state_key.and_then(|k| self.table.get(&k)) {
            return ExpandResult::Leaf(existing);
        }

        match state.status() {
            Status::Terminal(reward) => {
                let id = self.insert(state_key, NodeKind::Terminal, reward, std::iter::empty());
                ExpandResult::Leaf(id)
            }
            Status::Ongoing(player) => {
                state.chance_outcomes(&mut bufs.chances);
                if !bufs.chances.is_empty() {
                    let edges = bufs.chances.drain(..).map(Edge::new_chance);
                    let id = self.insert(state_key, NodeKind::Chance, 0.0, edges);
                    return ExpandResult::Chance(id);
                }

                bufs.actions.clear();
                state.legal_actions(&mut bufs.actions);
                ExpandResult::NeedsEval(player)
            }
        }
    }

    pub fn complete_expand(
        &mut self,
        eval: &NnOutput,
        bufs: &mut Bufs,
        player: Player,
        state_key: Option<u64>,
    ) -> NodeId {
        let priors = crate::utils::softmax_masked(&eval.policy_logits, &bufs.actions);
        let edges = bufs.actions.drain(..).zip(priors).map(|(action, prior)| {
            let logit = eval.policy_logits[action];
            Edge::new_decision(action, prior, logit)
        });
        self.insert(state_key, NodeKind::Decision(player), eval.value, edges)
    }

    // ── Backprop ─────────────────────────────────────────────────

    pub fn backprop(&mut self, path: &[(NodeId, usize)]) {
        for &(nid, eidx) in path.iter().rev() {
            self.edges_mut(nid)[eidx].visits += 1;

            let (sum_edge_visits, weighted_child_q) = {
                let edges = self.edges(nid);
                let sum = edges.iter().map(|e| e.visits).sum::<u32>();
                let mut wq = 0.0f32;
                for edge in edges {
                    if let Some(child_id) = edge.child {
                        wq += edge.visits as f32 * self[child_id].q;
                    }
                }
                (sum, wq)
            };

            let node = &mut self[nid];
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

        // Wrap old vecs in Option so we can .take() to move elements out
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

    // ── Navigation ───────────────────────────────────────────────

    /// Find the child reached by `action` from `node`, if any.
    pub fn child_for_action(&self, node: NodeId, action: usize) -> Option<NodeId> {
        self.edges(node)
            .iter()
            .find(|e| e.action == action)
            .and_then(|e| e.child)
    }

    /// Sample a chance edge proportional to priors.
    pub fn sample_chance_edge(&self, node: NodeId, rng: &mut fastrand::Rng) -> usize {
        let edges = self.edges(node);
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
            0.0,
            std::iter::once(Edge::new_chance((0, 1.0))),
        );
        let b = tree.insert(
            None,
            NodeKind::Terminal,
            0.0,
            std::iter::once(Edge::new_chance((0, 1.0))),
        );
        let c = tree.insert(None, NodeKind::Terminal, 0.0, std::iter::empty());

        // Patch child pointers
        tree.edges_mut(a)[0].child = Some(b);
        tree.edges_mut(b)[0].child = Some(c);

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
