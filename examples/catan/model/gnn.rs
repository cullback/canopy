//! GNN model for Catan.
//!
//! Preserves per-node features through message passing on the board graph,
//! producing per-node policy logits for settlement/city placement and
//! edge-gathered logits for road placement.

use std::sync::OnceLock;

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{relu, tanh};
use canopy2::nn::PolicyValueNet;

use crate::game::topology::Topology;

const NUM_NODES: usize = 54;
const NUM_EDGES: usize = 72;
const HIDDEN: usize = 128;
const GLOBAL_LEN: usize = 49;
const GLOBAL_HIDDEN: usize = 64;
const NUM_LAYERS: usize = 4;
const NODES_F: usize = 24;

// ── Static graph topology ────────────────────────────────────────────

/// Cached board graph data (node adjacency and edge endpoints).
/// The hexagonal grid structure is deterministic regardless of tile layout.
struct GraphData {
    /// Row-normalized adjacency matrix [54×54], flattened.
    adj_flat: Vec<f32>,
    /// Source node index for each edge.
    edge_src: [usize; NUM_EDGES],
    /// Destination node index for each edge.
    edge_dst: [usize; NUM_EDGES],
}

static GRAPH_DATA: OnceLock<GraphData> = OnceLock::new();

fn graph_data() -> &'static GraphData {
    GRAPH_DATA.get_or_init(|| {
        let topo = Topology::from_seed(0);

        // Row-normalized adjacency (no self-loops)
        let mut adj_flat = vec![0.0f32; NUM_NODES * NUM_NODES];
        for i in 0..NUM_NODES {
            let neighbors = &topo.nodes[i].adjacent_nodes;
            let degree = neighbors.len() as f32;
            for n in neighbors {
                adj_flat[i * NUM_NODES + n.0 as usize] = 1.0 / degree;
            }
        }

        // Edge endpoint indices
        let mut edge_src = [0usize; NUM_EDGES];
        let mut edge_dst = [0usize; NUM_EDGES];
        for e in 0..NUM_EDGES {
            let endpoints = topo.adj.edge_endpoints[e];
            let a = endpoints.trailing_zeros() as usize;
            let b = (endpoints & !(1u64 << a)).trailing_zeros() as usize;
            edge_src[e] = a;
            edge_dst[e] = b;
        }

        GraphData {
            adj_flat,
            edge_src,
            edge_dst,
        }
    })
}

// ── GNN Layer ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
struct GnnLayer<B: Backend> {
    norm: LayerNorm<B>,
    linear_self: Linear<B>,
    linear_msg: Linear<B>,
}

fn gnn_layer<B: Backend>(device: &B::Device) -> GnnLayer<B> {
    GnnLayer {
        norm: LayerNormConfig::new(HIDDEN).init(device),
        linear_self: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        linear_msg: LinearConfig::new(HIDDEN, HIDDEN).init(device),
    }
}

impl<B: Backend> GnnLayer<B> {
    /// Pre-norm residual GNN layer.
    ///
    /// h' = h + ReLU(Linear_self(LN(h)) + adj @ Linear_msg(LN(h)))
    fn forward(&self, h: Tensor<B, 3>, adj: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, nodes, hidden] = h.dims();

        // Pre-norm: flatten → LayerNorm → split into self/msg paths
        let h_norm = self
            .norm
            .forward(h.clone().reshape([batch * nodes, hidden]));

        let h_self = self
            .linear_self
            .forward(h_norm.clone())
            .reshape([batch, nodes, hidden]);

        let h_msg = self
            .linear_msg
            .forward(h_norm)
            .reshape([batch, nodes, hidden]);

        // Message aggregation: [1, 54, 54] @ [batch, 54, 128] → [batch, 54, 128]
        let h_agg = adj.matmul(h_msg);

        // Residual connection
        h + relu(h_self + h_agg)
    }
}

// ── GNN Model ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct CatanGnnModel<B: Backend> {
    // Input projections
    global_proj: Linear<B>,
    node_proj: Linear<B>,
    inject_proj: Linear<B>,
    inject_norm: LayerNorm<B>,

    // GNN trunk
    gnn_layers: Vec<GnnLayer<B>>,

    // Constant graph tensors (not trainable, built once in init)
    adj: Tensor<B, 3>,
    edge_src: Tensor<B, 1, Int>,
    edge_dst: Tensor<B, 1, Int>,

    // Policy head — settlement (per-node)
    policy_settlement: Linear<B>,
    // Policy head — road (edge endpoint sum → MLP)
    policy_road1: Linear<B>,
    policy_road2: Linear<B>,
    // Policy head — city (per-node)
    policy_city: Linear<B>,
    // Policy head — other (mean-pooled + global → MLP)
    policy_other1: Linear<B>,
    policy_other2: Linear<B>,

    // Value head (mean-pooled + global → MLP)
    value1: Linear<B>,
    value2: Linear<B>,
    value3: Linear<B>,
}

#[derive(Config, Debug)]
pub struct CatanGnnModelConfig {
    num_actions: usize,
}

impl CatanGnnModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CatanGnnModel<B> {
        let num_other = self.num_actions - NUM_NODES - NUM_EDGES - NUM_NODES;
        let graph = graph_data();

        // Build constant graph tensors once
        let adj = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(graph.adj_flat.clone(), [NUM_NODES, NUM_NODES]),
            device,
        )
        .unsqueeze::<3>(); // [1, 54, 54]

        let src_ints: Vec<i32> = graph.edge_src.iter().map(|&x| x as i32).collect();
        let dst_ints: Vec<i32> = graph.edge_dst.iter().map(|&x| x as i32).collect();
        let edge_src = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(src_ints, [NUM_EDGES]),
            device,
        );
        let edge_dst = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(dst_ints, [NUM_EDGES]),
            device,
        );

        CatanGnnModel {
            global_proj: LinearConfig::new(GLOBAL_LEN, GLOBAL_HIDDEN).init(device),
            node_proj: LinearConfig::new(NODES_F, HIDDEN).init(device),
            inject_proj: LinearConfig::new(HIDDEN + GLOBAL_HIDDEN, HIDDEN).init(device),
            inject_norm: LayerNormConfig::new(HIDDEN).init(device),

            gnn_layers: (0..NUM_LAYERS).map(|_| gnn_layer(device)).collect(),

            adj,
            edge_src,
            edge_dst,

            policy_settlement: LinearConfig::new(HIDDEN, 1).init(device),
            policy_road1: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            policy_road2: LinearConfig::new(HIDDEN, 1).init(device),
            policy_city: LinearConfig::new(HIDDEN, 1).init(device),
            policy_other1: LinearConfig::new(HIDDEN + GLOBAL_HIDDEN, HIDDEN).init(device),
            policy_other2: LinearConfig::new(HIDDEN, num_other).init(device),

            value1: LinearConfig::new(HIDDEN + GLOBAL_HIDDEN, HIDDEN).init(device),
            value2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            value3: LinearConfig::new(HIDDEN, 1).init(device),
        }
    }
}

impl<B: Backend> PolicyValueNet<B> for CatanGnnModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, _] = input.dims();

        // ── Split input ──────────────────────────────────────────────
        let global_raw = input.clone().narrow(1, 0, GLOBAL_LEN);
        let nodes_raw = input
            .narrow(1, GLOBAL_LEN, NUM_NODES * NODES_F)
            .reshape([batch, NUM_NODES, NODES_F]);

        // ── Project global: [batch, 49] → [batch, 64] ───────────────
        let global = relu(self.global_proj.forward(global_raw));

        // ── Project nodes: [batch, 54, 24] → [batch, 54, 128] ───────
        let nodes = relu(
            self.node_proj
                .forward(nodes_raw.reshape([batch * NUM_NODES, NODES_F])),
        )
        .reshape([batch, NUM_NODES, HIDDEN]);

        // ── Inject global into each node ─────────────────────────────
        // [batch, 64] → [batch, 1, 64] → [batch, 54, 64]
        let global_expanded = global
            .clone()
            .reshape([batch, 1, GLOBAL_HIDDEN])
            .repeat_dim(1, NUM_NODES);
        // cat(nodes, global) → [batch, 54, 192] → Linear + LayerNorm → [batch, 54, 128]
        let combined = Tensor::cat(vec![nodes, global_expanded], 2);
        let mut h = relu(
            self.inject_norm.forward(
                self.inject_proj
                    .forward(combined.reshape([batch * NUM_NODES, HIDDEN + GLOBAL_HIDDEN])),
            ),
        )
        .reshape([batch, NUM_NODES, HIDDEN]);

        // ── GNN trunk (4 layers) ─────────────────────────────────────
        // adj, edge_src, edge_dst are stored on the model (built once in init)
        for layer in &self.gnn_layers {
            h = layer.forward(h, self.adj.clone());
        }
        // h: [batch, 54, 128]

        // ── Policy: Settlement (actions 0-53) ────────────────────────
        // Per-node: [batch*54, 128] → [batch*54, 1] → [batch, 54]
        let settlement_logits = self
            .policy_settlement
            .forward(h.clone().reshape([batch * NUM_NODES, HIDDEN]))
            .reshape([batch, NUM_NODES]);

        // ── Policy: Road (actions 54-125) ────────────────────────────
        // Gather endpoint features and sum for direction invariance
        let h_src = h.clone().select(1, self.edge_src.clone());
        let h_dst = h.clone().select(1, self.edge_dst.clone());
        // Sum endpoints: [batch, 72, 128] (symmetric — invariant to edge direction)
        let edge_feats = h_src + h_dst;
        // MLP: [batch*72, 128] → [batch*72, 128] → [batch*72, 1] → [batch, 72]
        let road_logits = self
            .policy_road2
            .forward(relu(
                self.policy_road1
                    .forward(edge_feats.reshape([batch * NUM_EDGES, HIDDEN])),
            ))
            .reshape([batch, NUM_EDGES]);

        // ── Policy: City (actions 126-179) ───────────────────────────
        let city_logits = self
            .policy_city
            .forward(h.clone().reshape([batch * NUM_NODES, HIDDEN]))
            .reshape([batch, NUM_NODES]);

        // ── Mean pool + global bypass (shared by other-policy and value heads)
        // [batch, 54, 128] → [batch, 128], then cat with global → [batch, 192]
        let pooled = h.mean_dim(1).reshape([batch, HIDDEN]);
        let pooled_global = Tensor::cat(vec![pooled, global], 1);

        // ── Policy: Other (actions 180-248) ──────────────────────────
        let other_logits = self
            .policy_other2
            .forward(relu(self.policy_other1.forward(pooled_global.clone())));

        // ── Concatenate all policy logits ────────────────────────────
        let policy = Tensor::cat(
            vec![settlement_logits, road_logits, city_logits, other_logits],
            1,
        );

        // ── Value head ───────────────────────────────────────────────
        let v = relu(self.value1.forward(pooled_global));
        let v = relu(self.value2.forward(v));
        let value = tanh(self.value3.forward(v));

        (policy, value)
    }
}
