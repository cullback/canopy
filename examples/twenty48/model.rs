use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid};
use canopy::game::Game;
use canopy::nn::{ForwardOutput, PolicyValueNet};

use crate::game::Board;

const NUM_ACTIONS: usize = Board::NUM_ACTIONS;
const IN_CHANNELS: usize = 18;
const HIDDEN: usize = 64;
const NUM_BLOCKS: usize = 4;
const POOL_INNER: usize = 16;

// ---------------------------------------------------------------------------
// ResBlock with global-pooling gate (KataGo-style)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    pool_fc: Linear<B>,
    pool_gate: Linear<B>,
}

fn init_resblock<B: Backend>(device: &B::Device) -> ResBlock<B> {
    let conv_cfg = Conv2dConfig::new([HIDDEN, HIDDEN], [3, 3]).with_padding(PaddingConfig2d::Same);
    ResBlock {
        conv1: conv_cfg.clone().init(device),
        bn1: BatchNormConfig::new(HIDDEN).init(device),
        conv2: conv_cfg.init(device),
        bn2: BatchNormConfig::new(HIDDEN).init(device),
        pool_fc: LinearConfig::new(HIDDEN, POOL_INNER).init(device),
        pool_gate: LinearConfig::new(POOL_INNER, HIDDEN).init(device),
    }
}

impl<B: Backend> ResBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = relu(self.bn1.forward(self.conv1.forward(x.clone())));
        let h = self.bn2.forward(self.conv2.forward(h));

        // Global average pool → FC → ReLU → FC → sigmoid gate
        let [b, c, _, _] = h.dims();
        let pooled = h.clone().reshape([b, c, 16]).mean_dim(2).reshape([b, c]);
        let gate = sigmoid(self.pool_gate.forward(relu(self.pool_fc.forward(pooled))));
        let gate = gate.reshape([b, c, 1, 1]);

        // Channel-wise gating + residual
        relu(h.mul(gate) + x)
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Twenty48Model<B: Backend> {
    input_conv: Conv2d<B>,
    input_bn: BatchNorm<B>,
    blocks: Vec<ResBlock<B>>,
    policy_conv: Conv2d<B>,
    policy_bn: BatchNorm<B>,
    policy_fc: Linear<B>,
    value_conv: Conv2d<B>,
    value_bn: BatchNorm<B>,
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,
}

pub fn init_twenty48<B: Backend>(device: &B::Device) -> Twenty48Model<B> {
    Twenty48Model {
        input_conv: Conv2dConfig::new([IN_CHANNELS, HIDDEN], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device),
        input_bn: BatchNormConfig::new(HIDDEN).init(device),
        blocks: (0..NUM_BLOCKS).map(|_| init_resblock(device)).collect(),
        // Policy head: conv 1×1 → 2 channels → flatten(2*4*4=32) → FC → 4 actions
        policy_conv: Conv2dConfig::new([HIDDEN, 2], [1, 1]).init(device),
        policy_bn: BatchNormConfig::new(2).init(device),
        policy_fc: LinearConfig::new(2 * 16, NUM_ACTIONS).init(device),
        // Value head: conv 1×1 → 1 channel → flatten(1*4*4=16) → FC → 64 → FC → 3 (WDL)
        value_conv: Conv2dConfig::new([HIDDEN, 1], [1, 1]).init(device),
        value_bn: BatchNormConfig::new(1).init(device),
        value_fc1: LinearConfig::new(16, 32).init(device),
        value_fc2: LinearConfig::new(32, 3).init(device),
    }
}

impl<B: Backend> PolicyValueNet<B> for Twenty48Model<B> {
    fn forward(&self, input: Tensor<B, 2>) -> ForwardOutput<B> {
        let [batch, _] = input.dims();
        let x = input.reshape([batch, IN_CHANNELS, 4, 4]);

        // Stem
        let mut x = relu(self.input_bn.forward(self.input_conv.forward(x)));

        // Residual tower
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Policy head
        let p = relu(self.policy_bn.forward(self.policy_conv.forward(x.clone())));
        let policy = self.policy_fc.forward(p.reshape([batch, 2 * 16]));

        // Value head
        let v = relu(self.value_bn.forward(self.value_conv.forward(x)));
        let v = relu(self.value_fc1.forward(v.reshape([batch, 16])));
        let value = self.value_fc2.forward(v);

        ForwardOutput {
            policy_logits: policy,
            value,
            soft_policy_logits: None,
            aux_values: None,
        }
    }
}
