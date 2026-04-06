//! V1 heterogeneous GNN model for Catan.
//!
//! Same architecture as current nexus but with:
//! - NF=21 (no settle_legal, no longest_road_nodes)
//! - No edge features (EF=0)
//! - 4 GNN layers instead of 5
//! - Road policy head takes HIDDEN*2 (no edge features concatenated)

use std::sync::OnceLock;

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{relu, tanh};
use canopy::game::Game;
use canopy::nn::{ForwardOutput, PolicyValueNet};

use crate::game::state::GameState;
use crate::game::topology::Topology;

const NA: usize = GameState::NUM_ACTIONS;
const _: () = assert!(
    NA == 249,
    "action space changed — update split constants below"
);

const GL: usize = 121;
const TF: usize = 10;
const NF: usize = 21;
const NUM_TILES: usize = 19;
const NUM_NODES: usize = 54;
const NUM_EDGES: usize = 72;

const HIDDEN: usize = 256;
const GLOBAL_HIDDEN: usize = 96;
const NUM_LAYERS: usize = 4;

// Action layout (same as current, 249 total):
//   [settlement 0..54 | road 54..126 | city 126..180 | other_pre 180..205 | robber 205..224 | other_post 224..249]
const NUM_OTHER_PRE: usize = 25;
const NUM_OTHER_POST: usize = 25;
const NUM_OTHER: usize = NUM_OTHER_PRE + NUM_OTHER_POST;

// ── Static graph topology ────────────────────────────────────────────

struct NexusV1GraphData {
    node_adj_flat: Vec<f32>,
    tile_to_node_adj_flat: Vec<f32>,
    node_to_tile_adj_flat: Vec<f32>,
    edge_src: [usize; NUM_EDGES],
    edge_dst: [usize; NUM_EDGES],
}

static NEXUS_V1_GRAPH_DATA: OnceLock<NexusV1GraphData> = OnceLock::new();

fn nexus_v1_graph_data() -> &'static NexusV1GraphData {
    NEXUS_V1_GRAPH_DATA.get_or_init(|| {
        let topo = Topology::from_seed(0);

        let mut node_adj_flat = vec![0.0f32; NUM_NODES * NUM_NODES];
        for i in 0..NUM_NODES {
            let neighbors = &topo.nodes[i].adjacent_nodes;
            let degree = neighbors.len() as f32;
            for n in neighbors {
                node_adj_flat[i * NUM_NODES + n.0 as usize] = 1.0 / degree;
            }
        }

        let mut tile_to_node_adj_flat = vec![0.0f32; NUM_NODES * NUM_TILES];
        for i in 0..NUM_NODES {
            let adj_tiles = &topo.nodes[i].adjacent_tiles;
            let degree = adj_tiles.len() as f32;
            for &tid in adj_tiles {
                tile_to_node_adj_flat[i * NUM_TILES + tid.0 as usize] = 1.0 / degree;
            }
        }

        let mut node_to_tile_adj_flat = vec![0.0f32; NUM_TILES * NUM_NODES];
        for (j, tile) in topo.tiles.iter().enumerate() {
            for &nid in &tile.nodes {
                node_to_tile_adj_flat[j * NUM_NODES + nid.0 as usize] = 1.0 / 6.0;
            }
        }

        let mut edge_src = [0usize; NUM_EDGES];
        let mut edge_dst = [0usize; NUM_EDGES];
        for e in 0..NUM_EDGES {
            let endpoints = topo.adj.edge_endpoints[e];
            let a = endpoints.trailing_zeros() as usize;
            let b = (endpoints & !(1u64 << a)).trailing_zeros() as usize;
            edge_src[e] = a;
            edge_dst[e] = b;
        }

        NexusV1GraphData {
            node_adj_flat,
            tile_to_node_adj_flat,
            node_to_tile_adj_flat,
            edge_src,
            edge_dst,
        }
    })
}

// ── Heterogeneous GNN Layer ──────────────────────────────────────────

#[derive(Module, Debug)]
struct HeteroGnnLayerV1<B: Backend> {
    node_norm: LayerNorm<B>,
    tile_norm: LayerNorm<B>,
    linear_node_self: Linear<B>,
    linear_nn: Linear<B>,
    linear_tn: Linear<B>,
    linear_tile_self: Linear<B>,
    linear_nt: Linear<B>,
}

fn hetero_gnn_layer_v1<B: Backend>(device: &B::Device) -> HeteroGnnLayerV1<B> {
    HeteroGnnLayerV1 {
        node_norm: LayerNormConfig::new(HIDDEN).init(device),
        tile_norm: LayerNormConfig::new(HIDDEN).init(device),
        linear_node_self: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        linear_nn: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        linear_tn: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        linear_tile_self: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        linear_nt: LinearConfig::new(HIDDEN, HIDDEN).init(device),
    }
}

impl<B: Backend> HeteroGnnLayerV1<B> {
    fn forward(
        &self,
        node: Tensor<B, 3>,
        tile: Tensor<B, 3>,
        node_adj: Tensor<B, 3>,
        tile_to_node_adj: Tensor<B, 3>,
        node_to_tile_adj: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, num_nodes, hidden] = node.dims();
        let [_, num_tiles, _] = tile.dims();

        let node_norm = self
            .node_norm
            .forward(node.clone().reshape([batch * num_nodes, hidden]))
            .reshape([batch, num_nodes, hidden]);
        let tile_norm = self
            .tile_norm
            .forward(tile.clone().reshape([batch * num_tiles, hidden]))
            .reshape([batch, num_tiles, hidden]);

        let node_self = self
            .linear_node_self
            .forward(node_norm.clone().reshape([batch * num_nodes, hidden]))
            .reshape([batch, num_nodes, hidden]);
        let nn_msg = self
            .linear_nn
            .forward(node_norm.clone().reshape([batch * num_nodes, hidden]))
            .reshape([batch, num_nodes, hidden]);
        let nn_agg = node_adj.matmul(nn_msg);
        let tn_msg = self
            .linear_tn
            .forward(tile_norm.clone().reshape([batch * num_tiles, hidden]))
            .reshape([batch, num_tiles, hidden]);
        let tn_agg = tile_to_node_adj.matmul(tn_msg);
        let node_out = node + relu(node_self + nn_agg + tn_agg);

        let tile_self = self
            .linear_tile_self
            .forward(tile_norm.reshape([batch * num_tiles, hidden]))
            .reshape([batch, num_tiles, hidden]);
        let nt_msg = self
            .linear_nt
            .forward(node_norm.reshape([batch * num_nodes, hidden]))
            .reshape([batch, num_nodes, hidden]);
        let nt_agg = node_to_tile_adj.matmul(nt_msg);
        let tile_out = tile + relu(tile_self + nt_agg);

        (node_out, tile_out)
    }
}

// ── Nexus V1 Model ──────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct CatanNexusModelV1<B: Backend> {
    // Input projections
    global_proj: Linear<B>,
    tile_proj: Linear<B>,
    node_proj: Linear<B>,

    // Global injection
    node_inject_proj: Linear<B>,
    node_inject_norm: LayerNorm<B>,
    tile_inject_proj: Linear<B>,
    tile_inject_norm: LayerNorm<B>,

    // GNN trunk
    gnn_layers: Vec<HeteroGnnLayerV1<B>>,

    // Constant graph tensors
    node_adj: Tensor<B, 3>,
    tile_to_node_adj: Tensor<B, 3>,
    node_to_tile_adj: Tensor<B, 3>,
    edge_src: Tensor<B, 1, Int>,
    edge_dst: Tensor<B, 1, Int>,

    // Policy heads (2-layer each)
    policy_settlement: [Linear<B>; 2],
    policy_road: [Linear<B>; 2],
    policy_city: [Linear<B>; 2],
    policy_other: [Linear<B>; 2],
    policy_robber: [Linear<B>; 2],

    // Value head (3-layer)
    value: [Linear<B>; 3],

    // Auxiliary short-term value heads
    aux_value_hidden: Option<Linear<B>>,
    aux_value_out: Option<Linear<B>>,
}

pub fn init_nexus_v1<B: Backend>(device: &B::Device, num_aux_heads: usize) -> CatanNexusModelV1<B> {
    let graph = nexus_v1_graph_data();

    let node_adj = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(graph.node_adj_flat.clone(), [NUM_NODES, NUM_NODES]),
        device,
    )
    .unsqueeze::<3>();

    let tile_to_node_adj = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(graph.tile_to_node_adj_flat.clone(), [NUM_NODES, NUM_TILES]),
        device,
    )
    .unsqueeze::<3>();

    let node_to_tile_adj = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(graph.node_to_tile_adj_flat.clone(), [NUM_TILES, NUM_NODES]),
        device,
    )
    .unsqueeze::<3>();

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

    let pool_dim = HIDDEN + HIDDEN + GLOBAL_HIDDEN;

    CatanNexusModelV1 {
        global_proj: LinearConfig::new(GL, GLOBAL_HIDDEN).init(device),
        tile_proj: LinearConfig::new(TF, HIDDEN).init(device),
        node_proj: LinearConfig::new(NF, HIDDEN).init(device),

        node_inject_proj: LinearConfig::new(HIDDEN + GLOBAL_HIDDEN, HIDDEN).init(device),
        node_inject_norm: LayerNormConfig::new(HIDDEN).init(device),
        tile_inject_proj: LinearConfig::new(HIDDEN + GLOBAL_HIDDEN, HIDDEN).init(device),
        tile_inject_norm: LayerNormConfig::new(HIDDEN).init(device),

        gnn_layers: (0..NUM_LAYERS)
            .map(|_| hetero_gnn_layer_v1(device))
            .collect(),

        node_adj,
        tile_to_node_adj,
        node_to_tile_adj,
        edge_src,
        edge_dst,

        // 2-layer policy heads
        policy_settlement: [
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, 1).init(device),
        ],
        policy_road: [
            LinearConfig::new(HIDDEN * 2, HIDDEN).init(device), // no EF
            LinearConfig::new(HIDDEN, 1).init(device),
        ],
        policy_city: [
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, 1).init(device),
        ],
        policy_other: [
            LinearConfig::new(pool_dim, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, NUM_OTHER).init(device),
        ],
        policy_robber: [
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, 1).init(device),
        ],

        // 3-layer value head
        value: [
            LinearConfig::new(pool_dim, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, 3).init(device),
        ],

        aux_value_hidden: if num_aux_heads > 0 {
            Some(LinearConfig::new(pool_dim, HIDDEN).init(device))
        } else {
            None
        },
        aux_value_out: if num_aux_heads > 0 {
            Some(LinearConfig::new(HIDDEN, num_aux_heads).init(device))
        } else {
            None
        },
    }
}

impl<B: Backend> PolicyValueNet<B> for CatanNexusModelV1<B> {
    fn forward(&self, input: Tensor<B, 2>) -> ForwardOutput<B> {
        let [batch, _] = input.dims();

        // ── Split input: [global | tiles | nodes] (no edges) ────────
        let global_raw = input.clone().narrow(1, 0, GL);
        let tiles_raw = input
            .clone()
            .narrow(1, GL, NUM_TILES * TF)
            .reshape([batch, NUM_TILES, TF]);
        let nodes_raw = input
            .narrow(1, GL + NUM_TILES * TF, NUM_NODES * NF)
            .reshape([batch, NUM_NODES, NF]);

        // ── Project global ───────────────────────────────────────────
        let global = relu(self.global_proj.forward(global_raw));

        // ── Project tiles ──────────────────────────────────────────
        let tiles = relu(
            self.tile_proj
                .forward(tiles_raw.reshape([batch * NUM_TILES, TF])),
        )
        .reshape([batch, NUM_TILES, HIDDEN]);

        // ── Project nodes ──────────────────────────────────────────
        let nodes = relu(
            self.node_proj
                .forward(nodes_raw.reshape([batch * NUM_NODES, NF])),
        )
        .reshape([batch, NUM_NODES, HIDDEN]);

        // ── Inject global into nodes ─────────────────────────────────
        let global_for_nodes = global
            .clone()
            .reshape([batch, 1, GLOBAL_HIDDEN])
            .repeat_dim(1, NUM_NODES);
        let node_combined = Tensor::cat(vec![nodes, global_for_nodes], 2);
        let mut h_node = relu(
            self.node_inject_norm.forward(
                self.node_inject_proj
                    .forward(node_combined.reshape([batch * NUM_NODES, HIDDEN + GLOBAL_HIDDEN])),
            ),
        )
        .reshape([batch, NUM_NODES, HIDDEN]);

        // ── Inject global into tiles ─────────────────────────────────
        let global_for_tiles = global
            .clone()
            .reshape([batch, 1, GLOBAL_HIDDEN])
            .repeat_dim(1, NUM_TILES);
        let tile_combined = Tensor::cat(vec![tiles, global_for_tiles], 2);
        let mut h_tile = relu(
            self.tile_inject_norm.forward(
                self.tile_inject_proj
                    .forward(tile_combined.reshape([batch * NUM_TILES, HIDDEN + GLOBAL_HIDDEN])),
            ),
        )
        .reshape([batch, NUM_TILES, HIDDEN]);

        // ── GNN trunk ─────────────────────────────────────────────────
        for layer in &self.gnn_layers {
            let (n, t) = layer.forward(
                h_node,
                h_tile,
                self.node_adj.clone(),
                self.tile_to_node_adj.clone(),
                self.node_to_tile_adj.clone(),
            );
            h_node = n;
            h_tile = t;
        }

        // ── Policy: Settlement (actions 0-53) ────────────────────────
        let s = relu(
            self.policy_settlement[0].forward(h_node.clone().reshape([batch * NUM_NODES, HIDDEN])),
        );
        let settlement_logits = self.policy_settlement[1]
            .forward(s)
            .reshape([batch, NUM_NODES]);

        // ── Policy: Road (actions 54-125) ────────────────────────────
        // No edge features — just use node pair embeddings
        let h_src = h_node.clone().select(1, self.edge_src.clone());
        let h_dst = h_node.clone().select(1, self.edge_dst.clone());
        let edge_sum = h_src.clone() + h_dst.clone();
        let edge_diff = (h_src - h_dst).abs();
        let edge_feats = Tensor::cat(vec![edge_sum, edge_diff], 2);
        let r =
            relu(self.policy_road[0].forward(edge_feats.reshape([batch * NUM_EDGES, HIDDEN * 2])));
        let road_logits = self.policy_road[1].forward(r).reshape([batch, NUM_EDGES]);

        // ── Policy: City (actions 126-179) ───────────────────────────
        let c =
            relu(self.policy_city[0].forward(h_node.clone().reshape([batch * NUM_NODES, HIDDEN])));
        let city_logits = self.policy_city[1].forward(c).reshape([batch, NUM_NODES]);

        // ── Pooled features ─────────────────────────────────────────
        let node_pool = h_node.clone().mean_dim(1).reshape([batch, HIDDEN]);
        let tile_pool = h_tile.clone().mean_dim(1).reshape([batch, HIDDEN]);
        let pooled = Tensor::cat(vec![node_pool, tile_pool, global], 1);

        // ── Policy: Other (50 non-robber actions) ────────────────────
        let o = relu(self.policy_other[0].forward(pooled.clone()));
        let other_logits = self.policy_other[1].forward(o);
        let other_pre = other_logits.clone().narrow(1, 0, NUM_OTHER_PRE);
        let other_post = other_logits.narrow(1, NUM_OTHER_PRE, NUM_OTHER_POST);

        // ── Policy: Robber (actions 205-223) ─────────────────────────
        let rb = relu(
            self.policy_robber[0].forward(h_tile.clone().reshape([batch * NUM_TILES, HIDDEN])),
        );
        let robber_logits = self.policy_robber[1]
            .forward(rb)
            .reshape([batch, NUM_TILES]);

        // ── Concatenate policy logits ────────────────────────────────
        let policy = Tensor::cat(
            vec![
                settlement_logits,
                road_logits,
                city_logits,
                other_pre,
                robber_logits,
                other_post,
            ],
            1,
        );

        // ── Value head (3 layers) ───────────────────────────────────
        let mut v = pooled.clone();
        for layer in &self.value[..2] {
            v = relu(layer.forward(v));
        }
        let value = self.value[2].forward(v);

        // ── Auxiliary value heads ────────────────────────────────────
        let aux_values = if let (Some(aux_hidden), Some(aux_out)) =
            (&self.aux_value_hidden, &self.aux_value_out)
        {
            let h = relu(aux_hidden.forward(pooled));
            Some(tanh(aux_out.forward(h)))
        } else {
            None
        };

        ForwardOutput {
            policy_logits: policy,
            value,
            aux_values,
        }
    }
}
