use std::marker::PhantomData;

use burn::prelude::*;

use crate::eval::{Evaluation, Evaluator};
use crate::game::{Game, Status};

/// Encodes a game state into a fixed-size feature vector for neural network input.
///
/// The encoding is always from the current player's point of view: "me" first,
/// "opponent" second, so the NN sees a canonical perspective.
///
/// The `out` buffer is cleared before encoding. Callers reuse the same `Vec` to
/// avoid allocating on every evaluation.
pub trait StateEncoder<G: Game> {
    const FEATURE_SIZE: usize;
    fn encode(state: &G, out: &mut Vec<f32>);
}

/// A neural network with policy and value heads.
///
/// - Input: `[batch, features]` tensor (encoded by [`StateEncoder`])
/// - Policy output: `[batch, num_actions]` **raw logits** (pre-softmax).
/// - Value output: `[batch, 1]` in **`[-1, 1]`** (tanh-activated).
///   +1 = good for the perspective player encoded in the input.
pub trait PolicyValueNet<B: Backend> {
    fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>);
}

/// Wraps a [`PolicyValueNet`] for use as an [`Evaluator`].
///
/// Encodes the state using `E`, runs a forward pass, and returns raw policy
/// logits plus a value mapped to P1's perspective. MCTS handles softmax
/// masking in `complete_expand`.
///
/// `Clone` is shallow: burn model tensors are refcounted, so cloning
/// only increments Arc counts.
pub struct NeuralEvaluator<B: Backend, E, M> {
    model: M,
    device: B::Device,
    _marker: PhantomData<fn() -> (B, E)>,
}

impl<B: Backend, E, M: Clone> Clone for NeuralEvaluator<B, E, M>
where
    B::Device: Clone,
{
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            device: self.device.clone(),
            _marker: PhantomData,
        }
    }
}

impl<B: Backend, E, M> NeuralEvaluator<B, E, M> {
    pub fn new(model: M, device: B::Device) -> Self {
        Self {
            model,
            device,
            _marker: PhantomData,
        }
    }
}

impl<G, B, E, M> Evaluator<G> for NeuralEvaluator<B, E, M>
where
    G: Game,
    B: Backend,
    E: StateEncoder<G>,
    M: PolicyValueNet<B> + Send,
{
    fn evaluate(&self, state: &G, _rng: &mut fastrand::Rng) -> Evaluation {
        let current = match state.status() {
            Status::Ongoing(p) => p,
            Status::Terminal(reward) => return Evaluation::uniform(G::NUM_ACTIONS, reward),
        };

        let mut features = Vec::with_capacity(E::FEATURE_SIZE);
        E::encode(state, &mut features);

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(features, [1, E::FEATURE_SIZE]),
            &self.device,
        );

        let (policy_logits_tensor, value_tensor) = self.model.forward(input);

        let logits_data: Vec<f32> = policy_logits_tensor
            .into_data()
            .to_vec()
            .expect("policy tensor to_vec");

        let value_data: Vec<f32> = value_tensor
            .into_data()
            .to_vec()
            .expect("value tensor to_vec");

        // NN outputs value in [-1, 1] from perspective player's view.
        // Convert to P1's perspective by multiplying by current player's sign.
        let raw_value = value_data[0];
        let value = raw_value * current.sign();

        Evaluation {
            policy_logits: logits_data,
            value,
        }
    }

    fn evaluate_batch(&self, states: &[&G], rng: &mut fastrand::Rng) -> Vec<Evaluation> {
        if states.len() <= 1 {
            return states.iter().map(|s| self.evaluate(s, rng)).collect();
        }

        // Determine which states are terminal vs ongoing; collect signs for ongoing
        let mut signs = Vec::with_capacity(states.len());
        let mut nn_indices = Vec::new();
        let mut results: Vec<Option<Evaluation>> = (0..states.len()).map(|_| None).collect();

        for (i, state) in states.iter().enumerate() {
            match state.status() {
                Status::Terminal(reward) => {
                    results[i] = Some(Evaluation::uniform(G::NUM_ACTIONS, reward));
                }
                Status::Ongoing(player) => {
                    signs.push(player.sign());
                    nn_indices.push(i);
                }
            }
        }

        if nn_indices.is_empty() {
            return results.into_iter().map(|r| r.unwrap()).collect();
        }

        // Encode ongoing states and run inference
        let n = nn_indices.len();
        let mut features = Vec::with_capacity(n * E::FEATURE_SIZE);
        let mut buf = Vec::with_capacity(E::FEATURE_SIZE);
        for &i in &nn_indices {
            buf.clear();
            E::encode(states[i], &mut buf);
            features.extend_from_slice(&buf);
        }

        let (all_logits, all_values) = self.infer_features(&features, n, E::FEATURE_SIZE);

        // Split outputs per state and apply sign correction
        for (j, &i) in nn_indices.iter().enumerate() {
            let logits_start = j * G::NUM_ACTIONS;
            let logits = all_logits[logits_start..logits_start + G::NUM_ACTIONS].to_vec();
            let value = all_values[j] * signs[j];
            results[i] = Some(Evaluation {
                policy_logits: logits,
                value,
            });
        }

        results.into_iter().map(|r| r.unwrap()).collect()
    }

    fn infer_features(
        &self,
        features: &[f32],
        batch_size: usize,
        feature_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(features.to_vec(), [batch_size, feature_size]),
            &self.device,
        );

        let (policy_tensor, value_tensor) = self.model.forward(input);

        let logits: Vec<f32> = policy_tensor
            .into_data()
            .to_vec()
            .expect("policy tensor to_vec");
        let values: Vec<f32> = value_tensor
            .into_data()
            .to_vec()
            .expect("value tensor to_vec");

        (logits, values)
    }
}
