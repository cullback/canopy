use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use canopy2::nn::PolicyValueNet;

use super::*;

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

#[derive(Config, Debug)]
pub struct CatanModelConfig {
    num_actions: usize,
    nodes_f: usize,
    edges_f: usize,
    tiles_f: usize,
    ports_f: usize,
}

impl CatanModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CatanModel<B> {
        let sdim = stream_dim(self.tiles_f, self.ports_f);
        CatanModel {
            proj_global: LinearConfig::new(GLOBAL_LEN, GLOBAL_OUT).init(device),
            proj_tiles: if self.tiles_f > 0 {
                Some(LinearConfig::new(self.tiles_f, TILES_OUT).init(device))
            } else {
                None
            },
            proj_nodes: LinearConfig::new(self.nodes_f, NODES_OUT).init(device),
            proj_edges: LinearConfig::new(self.edges_f, EDGES_OUT).init(device),
            proj_ports: if self.ports_f > 0 {
                Some(LinearConfig::new(self.ports_f, PORTS_OUT).init(device))
            } else {
                None
            },
            trunk1: LinearConfig::new(sdim, 256).init(device),
            trunk2: LinearConfig::new(256, 256).init(device),
            policy_head: LinearConfig::new(256, self.num_actions).init(device),
            value_head1: LinearConfig::new(256, 64).init(device),
            value_head2: LinearConfig::new(64, 1).init(device),
        }
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
