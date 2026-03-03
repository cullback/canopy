use std::collections::VecDeque;
use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(u32);

struct NodeEntry<N> {
    weight: N,
    edge_start: u32,
    edge_count: u32,
}

pub struct DiGraph<N, E> {
    nodes: Vec<NodeEntry<N>>,
    edges: Vec<E>,
}

impl<N, E> Default for DiGraph<N, E> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
}

impl<N, E> DiGraph<N, E> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, weight: N, edges: impl IntoIterator<Item = E>) -> NodeId {
        let edge_start = self.edges.len() as u32;
        self.edges.extend(edges);
        let edge_count = self.edges.len() as u32 - edge_start;
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(NodeEntry {
            weight,
            edge_start,
            edge_count,
        });
        id
    }

    pub fn edges(&self, id: NodeId) -> &[E] {
        let entry = &self.nodes[id.0 as usize];
        &self.edges[entry.edge_start as usize..(entry.edge_start + entry.edge_count) as usize]
    }

    pub fn edges_mut(&mut self, id: NodeId) -> &mut [E] {
        let entry = &self.nodes[id.0 as usize];
        &mut self.edges[entry.edge_start as usize..(entry.edge_start + entry.edge_count) as usize]
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Compact the graph to only nodes reachable from `root`.
    ///
    /// `target` extracts the child `NodeId` from an edge (if any).
    /// `set_target` remaps child `NodeId`s in edges after compaction.
    ///
    /// Returns the new `NodeId` of the root (always `NodeId(0)`).
    pub fn retain_subtree(
        &mut self,
        root: NodeId,
        target: impl Fn(&E) -> Option<NodeId>,
        mut set_target: impl FnMut(&mut E, Option<NodeId>),
    ) -> NodeId {
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
                if let Some(child) = target(edge)
                    && old_to_new[child.0 as usize].is_none()
                {
                    old_to_new[child.0 as usize] = Some(NodeId(next_id));
                    next_id += 1;
                    queue.push_back(child);
                }
            }
        }

        // Wrap old vecs in Option so we can .take() to move elements out
        let mut old_nodes: Vec<Option<NodeEntry<N>>> = std::mem::take(&mut self.nodes)
            .into_iter()
            .map(Some)
            .collect();
        let mut old_edges: Vec<Option<E>> = std::mem::take(&mut self.edges)
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
                let old_child = target(&edge);
                let new_child = old_child.and_then(|c| old_to_new[c.0 as usize]);
                set_target(&mut edge, new_child);
                self.edges.push(edge);
            }
            let edge_count = self.edges.len() as u32 - edge_start;

            self.nodes.push(NodeEntry {
                weight: old_entry.weight,
                edge_start,
                edge_count,
            });
        }

        NodeId(0)
    }
}

impl<N, E> Index<NodeId> for DiGraph<N, E> {
    type Output = N;

    fn index(&self, id: NodeId) -> &N {
        &self.nodes[id.0 as usize].weight
    }
}

impl<N, E> IndexMut<NodeId> for DiGraph<N, E> {
    fn index_mut(&mut self, id: NodeId) -> &mut N {
        &mut self.nodes[id.0 as usize].weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestEdge {
        child: Option<NodeId>,
        label: &'static str,
    }

    #[test]
    fn retain_subtree_compacts() {
        let mut g: DiGraph<&str, TestEdge> = DiGraph::new();

        // A(0) -> B(1) -> C(2)
        let _a = g.add_node(
            "A",
            std::iter::once(TestEdge {
                child: None, // will patch
                label: "a_to_b",
            }),
        );
        let b = g.add_node(
            "B",
            std::iter::once(TestEdge {
                child: None, // will patch
                label: "b_to_c",
            }),
        );
        let c = g.add_node("C", std::iter::empty());

        // Patch child pointers
        g.edges_mut(NodeId(0))[0].child = Some(b);
        g.edges_mut(b)[0].child = Some(c);

        assert_eq!(g.node_count(), 3);

        // Reroot at B — only B and C should survive
        let new_root = g.retain_subtree(b, |e| e.child, |e, new_child| e.child = new_child);

        assert_eq!(new_root, NodeId(0));
        assert_eq!(g.node_count(), 2);
        assert_eq!(g[NodeId(0)], "B");
        assert_eq!(g[NodeId(1)], "C");

        let edges = g.edges(NodeId(0));
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].label, "b_to_c");
        assert_eq!(edges[0].child, Some(NodeId(1)));

        assert_eq!(g.edges(NodeId(1)).len(), 0);
    }
}
