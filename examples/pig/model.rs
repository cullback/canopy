use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use canopy2::nn::PolicyValueNet;

#[derive(Module, Debug)]
pub struct PigModel<B: Backend> {
    fc1: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

#[derive(Config, Debug)]
pub struct PigModelConfig {
    num_actions: usize,
    feature_size: usize,
}

impl PigModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PigModel<B> {
        PigModel {
            fc1: LinearConfig::new(self.feature_size, 32).init(device),
            policy_head: LinearConfig::new(32, self.num_actions).init(device),
            value_head: LinearConfig::new(32, 1).init(device),
        }
    }
}

impl<B: Backend> PolicyValueNet<B> for PigModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = burn::tensor::activation::relu(self.fc1.forward(input));

        let policy = self.policy_head.forward(x.clone());

        let v = self.value_head.forward(x);
        let value = burn::tensor::activation::tanh(v);

        (policy, value)
    }
}
