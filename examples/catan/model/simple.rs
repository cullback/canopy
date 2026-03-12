use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use canopy2::game::Game;
use canopy2::nn::PolicyValueNet;

use super::*;
use crate::encoder::{BasicEncoder, RichNodeEncoder};
use crate::game::state::GameState;

const NUM_ACTIONS: usize = GameState::NUM_ACTIONS;

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
    let sdim = stream_dim(BasicEncoder::TILES_F, BasicEncoder::PORTS_F);
    CatanModel {
        proj_global: LinearConfig::new(GLOBAL_LEN, GLOBAL_OUT).init(device),
        proj_tiles: Some(LinearConfig::new(BasicEncoder::TILES_F, TILES_OUT).init(device)),
        proj_nodes: LinearConfig::new(BasicEncoder::NODES_F, NODES_OUT).init(device),
        proj_edges: LinearConfig::new(BasicEncoder::EDGES_F, EDGES_OUT).init(device),
        proj_ports: Some(LinearConfig::new(BasicEncoder::PORTS_F, PORTS_OUT).init(device)),
        trunk1: LinearConfig::new(sdim, 256).init(device),
        trunk2: LinearConfig::new(256, 256).init(device),
        policy_head: LinearConfig::new(256, NUM_ACTIONS).init(device),
        value_head1: LinearConfig::new(256, 64).init(device),
        value_head2: LinearConfig::new(64, 1).init(device),
    }
}

pub fn init_simple_rich<B: Backend>(device: &B::Device) -> CatanModel<B> {
    let sdim = stream_dim(RichNodeEncoder::TILES_F, RichNodeEncoder::PORTS_F);
    CatanModel {
        proj_global: LinearConfig::new(GLOBAL_LEN, GLOBAL_OUT).init(device),
        proj_tiles: None,
        proj_nodes: LinearConfig::new(RichNodeEncoder::NODES_F, NODES_OUT).init(device),
        proj_edges: LinearConfig::new(RichNodeEncoder::EDGES_F, EDGES_OUT).init(device),
        proj_ports: None,
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
