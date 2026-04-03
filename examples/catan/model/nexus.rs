//! Heterogeneous GNN model for Catan.
//!
//! Nodes and tiles are first-class entities with cross-messaging, giving the
//! robber head direct per-tile embeddings and simplifying the node encoder.

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
const NF: usize = 25;
const EF: usize = 4;
const NUM_TILES: usize = 19;
const NUM_NODES: usize = 54;
const NUM_EDGES: usize = 72;

const HIDDEN: usize = 384;
const GLOBAL_HIDDEN: usize = 128;
const NUM_LAYERS: usize = 4;

// Action layout (must match action.rs ACTION_SPACE = 249):
//   [settlement 0..54 | road 54..126 | city 126..180 | other_pre 180..205 | robber 205..224 | other_post 224..249]
//
// other_pre (25): Roll(180), EndTurn(181), BuyDevCard(182), PlayKnight(183),
//                 PlayRoadBuilding(184), YearOfPlenty(185..200), Monopoly(200..205)
// robber    (19): MoveRobber(205..224)
// other_post(25): Discard(224..229), MaritimeTrade(229..249)
//
// The "other" MLP produces 50 logits split at index 25 to interleave robber
// logits from the per-tile head. If actions are added or reordered, these
// constants and the concat order in forward() must be updated together.
const NUM_OTHER_PRE: usize = 25;
const NUM_OTHER_POST: usize = 25;
const NUM_OTHER: usize = NUM_OTHER_PRE + NUM_OTHER_POST;

// ── Static graph topology ────────────────────────────────────────────

struct NexusGraphData {
    /// Row-normalized node-to-node adjacency [54×54], flattened.
    node_adj_flat: Vec<f32>,
    /// Row-normalized tile-to-node adjacency [54×19], flattened.
    /// node i receives from its 2-3 adjacent tiles.
    tile_to_node_adj_flat: Vec<f32>,
    /// Row-normalized node-to-tile adjacency [19×54], flattened.
    /// tile j receives from its 6 corner nodes (each = 1/6).
    node_to_tile_adj_flat: Vec<f32>,
    /// Source node index for each edge.
    edge_src: [usize; NUM_EDGES],
    /// Destination node index for each edge.
    edge_dst: [usize; NUM_EDGES],
}

static NEXUS_GRAPH_DATA: OnceLock<NexusGraphData> = OnceLock::new();

fn nexus_graph_data() -> &'static NexusGraphData {
    NEXUS_GRAPH_DATA.get_or_init(|| {
        let topo = Topology::from_seed(0);

        // Node-to-node adjacency (row-normalized, no self-loops)
        let mut node_adj_flat = vec![0.0f32; NUM_NODES * NUM_NODES];
        for i in 0..NUM_NODES {
            let neighbors = &topo.nodes[i].adjacent_nodes;
            let degree = neighbors.len() as f32;
            for n in neighbors {
                node_adj_flat[i * NUM_NODES + n.0 as usize] = 1.0 / degree;
            }
        }

        // Tile-to-node adjacency [54×19]: node i ← tiles adjacent to i
        let mut tile_to_node_adj_flat = vec![0.0f32; NUM_NODES * NUM_TILES];
        for i in 0..NUM_NODES {
            let adj_tiles = &topo.nodes[i].adjacent_tiles;
            let degree = adj_tiles.len() as f32;
            for &tid in adj_tiles {
                tile_to_node_adj_flat[i * NUM_TILES + tid.0 as usize] = 1.0 / degree;
            }
        }

        // Node-to-tile adjacency [19×54]: tile j ← 6 corner nodes
        let mut node_to_tile_adj_flat = vec![0.0f32; NUM_TILES * NUM_NODES];
        for (j, tile) in topo.tiles.iter().enumerate() {
            for &nid in &tile.nodes {
                node_to_tile_adj_flat[j * NUM_NODES + nid.0 as usize] = 1.0 / 6.0;
            }
        }

        // Edge endpoints
        let mut edge_src = [0usize; NUM_EDGES];
        let mut edge_dst = [0usize; NUM_EDGES];
        for e in 0..NUM_EDGES {
            let endpoints = topo.adj.edge_endpoints[e];
            let a = endpoints.trailing_zeros() as usize;
            let b = (endpoints & !(1u64 << a)).trailing_zeros() as usize;
            edge_src[e] = a;
            edge_dst[e] = b;
        }

        NexusGraphData {
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
struct HeteroGnnLayer<B: Backend> {
    node_norm: LayerNorm<B>,
    tile_norm: LayerNorm<B>,
    linear_node_self: Linear<B>,
    linear_nn: Linear<B>,
    linear_tn: Linear<B>,
    linear_tile_self: Linear<B>,
    linear_nt: Linear<B>,
}

fn hetero_gnn_layer<B: Backend>(device: &B::Device) -> HeteroGnnLayer<B> {
    HeteroGnnLayer {
        node_norm: LayerNormConfig::new(HIDDEN).init(device),
        tile_norm: LayerNormConfig::new(HIDDEN).init(device),
        linear_node_self: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        linear_nn: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        linear_tn: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        linear_tile_self: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        linear_nt: LinearConfig::new(HIDDEN, HIDDEN).init(device),
    }
}

impl<B: Backend> HeteroGnnLayer<B> {
    /// Pre-norm residual heterogeneous GNN layer.
    ///
    /// ```text
    /// node_norm = LN(node);  tile_norm = LN(tile)
    /// node = node + ReLU(Linear_self(node_norm) + node_adj @ Linear_nn(node_norm) + tile_to_node_adj @ Linear_tn(tile_norm))
    /// tile = tile + ReLU(Linear_tile_self(tile_norm) + node_to_tile_adj @ Linear_nt(node_norm))
    /// ```
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

        // Pre-norm
        let node_norm = self
            .node_norm
            .forward(node.clone().reshape([batch * num_nodes, hidden]))
            .reshape([batch, num_nodes, hidden]);
        let tile_norm = self
            .tile_norm
            .forward(tile.clone().reshape([batch * num_tiles, hidden]))
            .reshape([batch, num_tiles, hidden]);

        // Node update: self + node-to-node messages + tile-to-node messages
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

        // Tile update: self + node-to-tile messages
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

// ── Nexus Model ──────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct CatanNexusModel<B: Backend> {
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
    gnn_layers: Vec<HeteroGnnLayer<B>>,

    // Constant graph tensors
    node_adj: Tensor<B, 3>,
    tile_to_node_adj: Tensor<B, 3>,
    node_to_tile_adj: Tensor<B, 3>,
    edge_src: Tensor<B, 1, Int>,
    edge_dst: Tensor<B, 1, Int>,

    // Policy heads
    policy_settlement: [Linear<B>; 4],
    policy_road: [Linear<B>; 4],
    policy_city: [Linear<B>; 4],
    policy_other: [Linear<B>; 4],
    policy_robber: [Linear<B>; 4],

    // Value head
    value: [Linear<B>; 4],

    // Auxiliary short-term value heads (None if num_aux_heads == 0)
    aux_value_hidden: Option<Linear<B>>,
    aux_value_out: Option<Linear<B>>,
}

pub fn init_nexus<B: Backend>(device: &B::Device, num_aux_heads: usize) -> CatanNexusModel<B> {
    let graph = nexus_graph_data();

    // Build constant graph tensors
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

    CatanNexusModel {
        global_proj: LinearConfig::new(GL, GLOBAL_HIDDEN).init(device),
        tile_proj: LinearConfig::new(TF, HIDDEN).init(device),
        node_proj: LinearConfig::new(NF, HIDDEN).init(device),

        node_inject_proj: LinearConfig::new(HIDDEN + GLOBAL_HIDDEN, HIDDEN).init(device),
        node_inject_norm: LayerNormConfig::new(HIDDEN).init(device),
        tile_inject_proj: LinearConfig::new(HIDDEN + GLOBAL_HIDDEN, HIDDEN).init(device),
        tile_inject_norm: LayerNormConfig::new(HIDDEN).init(device),

        gnn_layers: (0..NUM_LAYERS).map(|_| hetero_gnn_layer(device)).collect(),

        node_adj,
        tile_to_node_adj,
        node_to_tile_adj,
        edge_src,
        edge_dst,

        policy_settlement: [
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, 1).init(device),
        ],
        policy_road: [
            LinearConfig::new(HIDDEN * 2 + EF, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, 1).init(device),
        ],
        policy_city: [
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, 1).init(device),
        ],
        policy_other: [
            LinearConfig::new(pool_dim, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, NUM_OTHER).init(device),
        ],
        policy_robber: [
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, 1).init(device),
        ],

        value: [
            LinearConfig::new(pool_dim, HIDDEN).init(device),
            LinearConfig::new(HIDDEN, HIDDEN).init(device),
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

impl<B: Backend> PolicyValueNet<B> for CatanNexusModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> ForwardOutput<B> {
        let [batch, _] = input.dims();

        // ── Split input: [global | tiles | nodes | edges] ───────────
        let global_raw = input.clone().narrow(1, 0, GL);
        let tiles_raw = input
            .clone()
            .narrow(1, GL, NUM_TILES * TF)
            .reshape([batch, NUM_TILES, TF]);
        let nodes_raw = input
            .clone()
            .narrow(1, GL + NUM_TILES * TF, NUM_NODES * NF)
            .reshape([batch, NUM_NODES, NF]);
        let edges_raw = input
            .narrow(1, GL + NUM_TILES * TF + NUM_NODES * NF, NUM_EDGES * EF)
            .reshape([batch, NUM_EDGES, EF]);

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
        // h_node: [batch, 54, HIDDEN], h_tile: [batch, 19, HIDDEN]

        // ── Policy: Settlement (actions 0-53) ────────────────────────
        let mut s = h_node.clone().reshape([batch * NUM_NODES, HIDDEN]);
        for layer in &self.policy_settlement[..3] {
            s = relu(layer.forward(s));
        }
        let settlement_logits = self.policy_settlement[3]
            .forward(s)
            .reshape([batch, NUM_NODES]);

        // ── Policy: Road (actions 54-125) ────────────────────────────
        let h_src = h_node.clone().select(1, self.edge_src.clone());
        let h_dst = h_node.clone().select(1, self.edge_dst.clone());
        let edge_sum = h_src.clone() + h_dst.clone();
        let edge_diff = (h_src - h_dst).abs();
        let edge_feats = Tensor::cat(vec![edge_sum, edge_diff, edges_raw], 2);
        let mut r = edge_feats.reshape([batch * NUM_EDGES, HIDDEN * 2 + EF]);
        for layer in &self.policy_road[..3] {
            r = relu(layer.forward(r));
        }
        let road_logits = self.policy_road[3].forward(r).reshape([batch, NUM_EDGES]);

        // ── Policy: City (actions 126-179) ───────────────────────────
        let mut c = h_node.clone().reshape([batch * NUM_NODES, HIDDEN]);
        for layer in &self.policy_city[..3] {
            c = relu(layer.forward(c));
        }
        let city_logits = self.policy_city[3].forward(c).reshape([batch, NUM_NODES]);

        // ── Pooled features (shared by other-policy and value heads) ─
        let node_pool = h_node.clone().mean_dim(1).reshape([batch, HIDDEN]);
        let tile_pool = h_tile.clone().mean_dim(1).reshape([batch, HIDDEN]);
        let pooled = Tensor::cat(vec![node_pool, tile_pool, global], 1);

        // ── Policy: Other (50 non-robber actions) ────────────────────
        let mut o = pooled.clone();
        for layer in &self.policy_other[..3] {
            o = relu(layer.forward(o));
        }
        let other_logits = self.policy_other[3].forward(o);
        // Split into pre-robber [25] and post-robber [25]
        let other_pre = other_logits.clone().narrow(1, 0, NUM_OTHER_PRE);
        let other_post = other_logits.narrow(1, NUM_OTHER_PRE, NUM_OTHER_POST);

        // ── Policy: Robber (actions 205-223) ─────────────────────────
        let mut rb = h_tile.clone().reshape([batch * NUM_TILES, HIDDEN]);
        for layer in &self.policy_robber[..3] {
            rb = relu(layer.forward(rb));
        }
        let robber_logits = self.policy_robber[3]
            .forward(rb)
            .reshape([batch, NUM_TILES]);

        // ── Concatenate: [settlement(54)|road(72)|city(54)|other_pre(25)|robber(19)|other_post(25)]
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

        // ── Value head ───────────────────────────────────────────────
        let mut v = pooled.clone();
        for layer in &self.value[..3] {
            v = relu(layer.forward(v));
        }
        let value = self.value[3].forward(v);

        // ── Auxiliary value heads ────────────────────────────────────
        let aux_values = if let (Some(aux_hidden), Some(aux_out)) =
            (&self.aux_value_hidden, &self.aux_value_out)
        {
            let h = relu(aux_hidden.forward(pooled));
            Some(tanh(aux_out.forward(h))) // [batch, num_aux]
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
