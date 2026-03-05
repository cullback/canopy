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
        let h = relu(self.norm1.forward(self.linear1.forward(x)));
        let h = self.norm2.forward(self.linear2.forward(h));
        relu(h + residual)
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
    proj_tiles: Linear<B>,
    proj_nodes: Linear<B>,
    proj_edges: Linear<B>,
    proj_ports: Linear<B>,
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
}

impl CatanResModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CatanResModel<B> {
        CatanResModel {
            proj_global: LinearConfig::new(GLOBAL_LEN, 64).init(device),
            proj_tiles: LinearConfig::new(TILES_F, 32).init(device),
            proj_nodes: LinearConfig::new(NODES_F, 32).init(device),
            proj_edges: LinearConfig::new(EDGES_F, 16).init(device),
            proj_ports: LinearConfig::new(PORTS_F, 16).init(device),
            input_linear: LinearConfig::new(STREAM_DIM, 256).init(device),
            input_norm: LayerNormConfig::new(256).init(device),
            blocks: (0..6).map(|_| res_block(256, device)).collect(),
            policy_head1: LinearConfig::new(256, 256).init(device),
            policy_head2: LinearConfig::new(256, self.num_actions).init(device),
            value_head1: LinearConfig::new(256, 64).init(device),
            value_head2: LinearConfig::new(64, 1).init(device),
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
