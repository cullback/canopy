use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use canopy::game::Game;
use canopy::nn::PolicyValueNet;

use crate::encoder::Twenty48Encoder;
use crate::game::Board;

const NUM_ACTIONS: usize = Board::NUM_ACTIONS;
const FEATURE_SIZE: usize = Twenty48Encoder::FEATURE_SIZE;
const HIDDEN: usize = 128;

#[derive(Module, Debug)]
pub struct Twenty48Model<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

pub fn init_twenty48<B: Backend>(device: &B::Device) -> Twenty48Model<B> {
    Twenty48Model {
        fc1: LinearConfig::new(FEATURE_SIZE, HIDDEN).init(device),
        fc2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
        policy_head: LinearConfig::new(HIDDEN, NUM_ACTIONS).init(device),
        value_head: LinearConfig::new(HIDDEN, 1).init(device),
    }
}

impl<B: Backend> PolicyValueNet<B> for Twenty48Model<B> {
    fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = burn::tensor::activation::relu(self.fc1.forward(input));
        let x = burn::tensor::activation::relu(self.fc2.forward(x));

        let policy = self.policy_head.forward(x.clone());

        let v = self.value_head.forward(x);
        let value = burn::tensor::activation::tanh(v);

        (policy, value)
    }
}
