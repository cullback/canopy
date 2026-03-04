use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use canopy2::nn::PolicyValueNet;

/// Dual-head neural network: shared trunk -> policy head + value head.
///
/// Architecture: input -> 512 -> ReLU -> 256 -> ReLU -> split:
/// - Policy: 256 -> num_actions (logits, no activation)
/// - Value: 256 -> 64 -> ReLU -> 1 -> tanh (maps to [-1, 1])
#[derive(Module, Debug)]
pub struct CatanModel<B: Backend> {
    pub(crate) trunk1: Linear<B>,
    pub(crate) trunk2: Linear<B>,
    pub(crate) policy_head: Linear<B>,
    pub(crate) value_head1: Linear<B>,
    pub(crate) value_head2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct CatanModelConfig {
    input_size: usize,
    num_actions: usize,
    #[config(default = "512")]
    trunk_size: usize,
    #[config(default = "256")]
    hidden_size: usize,
}

impl CatanModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CatanModel<B> {
        CatanModel {
            trunk1: LinearConfig::new(self.input_size, self.trunk_size).init(device),
            trunk2: LinearConfig::new(self.trunk_size, self.hidden_size).init(device),
            policy_head: LinearConfig::new(self.hidden_size, self.num_actions).init(device),
            value_head1: LinearConfig::new(self.hidden_size, 64).init(device),
            value_head2: LinearConfig::new(64, 1).init(device),
        }
    }
}

impl<B: Backend> PolicyValueNet<B> for CatanModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.trunk1.forward(input);
        let x = burn::tensor::activation::relu(x);
        let x = self.trunk2.forward(x);
        let x = burn::tensor::activation::relu(x);

        let policy = self.policy_head.forward(x.clone());

        let v = self.value_head1.forward(x);
        let v = burn::tensor::activation::relu(v);
        let v = self.value_head2.forward(v);
        let value = burn::tensor::activation::tanh(v);

        (policy, value)
    }
}
