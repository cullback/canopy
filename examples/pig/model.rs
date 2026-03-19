use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use canopy::game::Game;
use canopy::nn::PolicyValueNet;

use crate::encoder::PigEncoder;
use crate::game::PigGame;

const NUM_ACTIONS: usize = PigGame::NUM_ACTIONS;
const FEATURE_SIZE: usize = PigEncoder::FEATURE_SIZE;

#[derive(Module, Debug)]
pub struct PigModel<B: Backend> {
    fc1: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

pub fn init_pig<B: Backend>(device: &B::Device) -> PigModel<B> {
    PigModel {
        fc1: LinearConfig::new(FEATURE_SIZE, 32).init(device),
        policy_head: LinearConfig::new(32, NUM_ACTIONS).init(device),
        value_head: LinearConfig::new(32, 1).init(device),
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
