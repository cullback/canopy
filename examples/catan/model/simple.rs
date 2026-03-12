use burn::nn::{Linear, LinearConfig};
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

#[derive(Module, Debug)]
pub struct CatanModel<B: Backend> {
    proj_global: Linear<B>,
    proj_tiles: Option<Linear<B>>,
    proj_nodes: Linear<B>,
    proj_edges: Linear<B>,
    proj_ports: Option<Linear<B>>,
    trunk1: Linear<B>,
    trunk2: Linear<B>,
    policy_head: Linear<B>,
    value_head1: Linear<B>,
    value_head2: Linear<B>,
}

pub fn init_simple<B: Backend>(device: &B::Device) -> CatanModel<B> {
    init_simple_with(NODES_F, EDGES_F, TILES_F, PORTS_F, device)
}

pub fn init_simple_with<B: Backend>(
    nodes_f: usize,
    edges_f: usize,
    tiles_f: usize,
    ports_f: usize,
    device: &B::Device,
) -> CatanModel<B> {
    let sdim = stream_dim(tiles_f, ports_f);
    CatanModel {
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
        trunk1: LinearConfig::new(sdim, 256).init(device),
        trunk2: LinearConfig::new(256, 256).init(device),
        policy_head: LinearConfig::new(256, NUM_ACTIONS).init(device),
        value_head1: LinearConfig::new(256, 64).init(device),
        value_head2: LinearConfig::new(64, 1).init(device),
    }
}

impl<B: Backend> PolicyValueNet<B> for CatanModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = project_streams(
            &input,
            &self.proj_global,
            &self.proj_tiles,
            &self.proj_nodes,
            &self.proj_edges,
            &self.proj_ports,
        );
        let x = relu(self.trunk1.forward(x));
        let x = relu(self.trunk2.forward(x));

        let policy = self.policy_head.forward(x.clone());

        let v = relu(self.value_head1.forward(x));
        let v = self.value_head2.forward(v);
        let value = burn::tensor::activation::tanh(v);

        (policy, value)
    }
}
