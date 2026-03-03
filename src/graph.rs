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
