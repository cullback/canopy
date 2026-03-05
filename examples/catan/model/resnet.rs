use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use canopy2::nn::PolicyValueNet;

use super::*;

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
    value_head1: Linear<B>,
    value_head2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct CatanResModelConfig {
    num_actions: usize,
    nodes_f: usize,
    edges_f: usize,
    tiles_f: usize,
    ports_f: usize,
}

impl CatanResModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CatanResModel<B> {
        let sdim = stream_dim(self.tiles_f, self.ports_f);
        CatanResModel {
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
            input_linear: LinearConfig::new(sdim, 256).init(device),
            input_norm: LayerNormConfig::new(256).init(device),
            blocks: (0..6).map(|_| res_block(256, device)).collect(),
            policy_head1: LinearConfig::new(256, 256).init(device),
            policy_head2: LinearConfig::new(256, self.num_actions).init(device),
            value_head1: LinearConfig::new(256, 128).init(device),
            value_head2: LinearConfig::new(128, 1).init(device),
        }
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

        let v = relu(self.value_head1.forward(x));
        let v = self.value_head2.forward(v);
        let value = burn::tensor::activation::tanh(v);

        (policy, value)
    }
}
