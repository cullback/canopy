use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use canopy::game::Game;
use canopy::nn::{ForwardOutput, PolicyValueNet};

use crate::encoder::TenureEncoder;
use crate::game::TenureGame;

const NUM_ACTIONS: usize = TenureGame::NUM_ACTIONS;
const FEATURE_SIZE: usize = TenureEncoder::FEATURE_SIZE;
const HIDDEN: usize = 128;

#[derive(Module, Debug)]
pub struct TenureModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

pub fn init_tenure<B: Backend>(device: &B::Device) -> TenureModel<B> {
    TenureModel {
        fc1: LinearConfig::new(FEATURE_SIZE, HIDDEN).init(device),
        fc2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        policy_head: LinearConfig::new(HIDDEN, NUM_ACTIONS).init(device),
        value_head: LinearConfig::new(HIDDEN, 3).init(device),
    }
}

impl<B: Backend> PolicyValueNet<B> for TenureModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> ForwardOutput<B> {
        let x = burn::tensor::activation::relu(self.fc1.forward(input));
        let x = burn::tensor::activation::relu(self.fc2.forward(x));

        let policy = self.policy_head.forward(x.clone());

        let value = self.value_head.forward(x);

        ForwardOutput {
            policy_logits: policy,
            value,
            soft_policy_logits: None,
            aux_values: None,
        }
    }
}
