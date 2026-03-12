use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use canopy2::game::Game;
use canopy2::nn::PolicyValueNet;

use super::*;
use crate::encoder::BasicEncoder;
use crate::game::state::GameState;

const NUM_ACTIONS: usize = GameState::NUM_ACTIONS;
const NODES_F: usize = BasicEncoder::NODES_F;
const EDGES_F: usize = BasicEncoder::EDGES_F;
const TILES_F: usize = BasicEncoder::TILES_F;
const PORTS_F: usize = BasicEncoder::PORTS_F;
const TRUNK_DIM: usize = 384;

#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    linear1: Linear<B>,
    norm1: LayerNorm<B>,
    linear2: Linear<B>,
    norm2: LayerNorm<B>,
}

impl<B: Backend> ResBlock<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let residual = x.clone();
        let h = self.linear1.forward(relu(self.norm1.forward(x)));
        let h = self.linear2.forward(relu(self.norm2.forward(h)));
        h + residual
    }
}

fn res_block<B: Backend>(dim: usize, device: &B::Device) -> ResBlock<B> {
    ResBlock {
        linear1: LinearConfig::new(dim, dim).init(device),
        norm1: LayerNormConfig::new(dim).init(device),
        linear2: LinearConfig::new(dim, dim).init(device),
        norm2: LayerNormConfig::new(dim).init(device),
    }
}

#[derive(Module, Debug)]
pub struct CatanResModel<B: Backend> {
    proj_global: Linear<B>,
    proj_tiles: Option<Linear<B>>,
    proj_nodes: Linear<B>,
    proj_edges: Linear<B>,
    proj_ports: Option<Linear<B>>,
    input_linear: Linear<B>,
    input_norm: LayerNorm<B>,
    blocks: Vec<ResBlock<B>>,
    policy_head1: Linear<B>,
    policy_head2: Linear<B>,
    value_block: ResBlock<B>,
    value_head1: Linear<B>,
    value_head2: Linear<B>,
    value_head3: Linear<B>,
}

pub fn init_resnet<B: Backend>(device: &B::Device) -> CatanResModel<B> {
    init_resnet_with(NODES_F, EDGES_F, TILES_F, PORTS_F, device)
}

pub fn init_resnet_with<B: Backend>(
    nodes_f: usize,
    edges_f: usize,
    tiles_f: usize,
    ports_f: usize,
    device: &B::Device,
) -> CatanResModel<B> {
    let sdim = stream_dim(tiles_f, ports_f);
    CatanResModel {
        proj_global: LinearConfig::new(GLOBAL_LEN, GLOBAL_OUT).init(device),
        proj_tiles: if tiles_f > 0 {
            Some(LinearConfig::new(tiles_f, TILES_OUT).init(device))
        } else {
            None
        },
        proj_nodes: LinearConfig::new(nodes_f, NODES_OUT).init(device),
        proj_edges: LinearConfig::new(edges_f, EDGES_OUT).init(device),
        proj_ports: if ports_f > 0 {
            Some(LinearConfig::new(ports_f, PORTS_OUT).init(device))
        } else {
            None
        },
        input_linear: LinearConfig::new(sdim, TRUNK_DIM).init(device),
        input_norm: LayerNormConfig::new(TRUNK_DIM).init(device),
        blocks: (0..6).map(|_| res_block(TRUNK_DIM, device)).collect(),
        policy_head1: LinearConfig::new(TRUNK_DIM, TRUNK_DIM).init(device),
        policy_head2: LinearConfig::new(TRUNK_DIM, NUM_ACTIONS).init(device),
        value_block: res_block(TRUNK_DIM, device),
        value_head1: LinearConfig::new(TRUNK_DIM, 128).init(device),
        value_head2: LinearConfig::new(128, 128).init(device),
        value_head3: LinearConfig::new(128, 1).init(device),
    }
}

impl<B: Backend> PolicyValueNet<B> for CatanResModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = project_streams(
            &input,
            &self.proj_global,
            &self.proj_tiles,
            &self.proj_nodes,
            &self.proj_edges,
            &self.proj_ports,
        );
        let mut x = relu(self.input_norm.forward(self.input_linear.forward(x)));

        for block in &self.blocks {
            x = block.forward(x);
        }

        let policy = relu(self.policy_head1.forward(x.clone()));
        let policy = self.policy_head2.forward(policy);

        let v = self.value_block.forward(x);
        let v = relu(self.value_head1.forward(v));
        let v = relu(self.value_head2.forward(v));
        let v = self.value_head3.forward(v);
        let value = burn::tensor::activation::tanh(v);

        (policy, value)
    }
}
