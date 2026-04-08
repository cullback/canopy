use std::sync::Arc;

use burn::prelude::*;

use crate::eval::{Evaluation, Evaluator, flip_wdl};
use crate::game::{Game, Status};

/// Encodes a game state into a fixed-size feature vector for neural network input.
///
/// The encoding is always from the current player's point of view: "me" first,
/// "opponent" second, so the NN sees a canonical perspective.
///
/// The `out` buffer is cleared before encoding. Callers reuse the same `Vec` to
/// avoid allocating on every evaluation.
pub trait StateEncoder<G: Game>: Send + Sync {
    fn feature_size(&self) -> usize;
    fn encode(&self, state: &G, out: &mut Vec<f32>);
}

/// Output of a forward pass through a [`PolicyValueNet`].
pub struct ForwardOutput<B: Backend> {
    /// Hard policy logits `[batch, num_actions]` (pre-softmax).
    pub policy_logits: Tensor<B, 2>,
    /// WDL logits `[batch, 3]` (pre-softmax). Softmax → `[win, draw, loss]`
    /// probabilities from the current player's perspective.
    pub value: Tensor<B, 2>,
    /// Soft policy logits `[batch, num_actions]` — `None` if no soft policy head.
    /// Trained against `policy_target^(1/T)` (renormalized) for a temperature T.
    pub soft_policy_logits: Option<Tensor<B, 2>>,
    /// Auxiliary short-term value predictions `[batch, num_aux]` — `None` if no aux heads.
    pub aux_values: Option<Tensor<B, 2>>,
}

/// A neural network with policy and value heads.
///
/// - Input: `[batch, features]` tensor (encoded by [`StateEncoder`])
/// - Policy output: `[batch, num_actions]` **raw logits** (pre-softmax).
/// - Value output: `[batch, 3]` **WDL logits** (pre-softmax).
///   Softmax → `[win, draw, loss]` from the current player's perspective.
pub trait PolicyValueNet<B: Backend> {
    fn forward(&self, input: Tensor<B, 2>) -> ForwardOutput<B>;
}

/// Wraps a [`PolicyValueNet`] for use as an [`Evaluator`].
///
/// Encodes the state using the stored encoder, runs a forward pass, and returns
/// raw policy logits plus a value mapped to P1's perspective. MCTS handles
/// softmax masking in `complete_expand`.
///
/// `Clone` is shallow: burn model tensors are refcounted, so cloning
/// only increments Arc counts.
pub struct NeuralEvaluator<G, B: Backend, M> {
    encoder: Arc<dyn StateEncoder<G>>,
    model: M,
    device: B::Device,
}

impl<G, B: Backend, M: Clone> Clone for NeuralEvaluator<G, B, M>
where
    B::Device: Clone,
{
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            model: self.model.clone(),
            device: self.device.clone(),
        }
    }
}

impl<G, B: Backend, M> NeuralEvaluator<G, B, M> {
    pub fn new(encoder: Arc<dyn StateEncoder<G>>, model: M, device: B::Device) -> Self {
        Self {
            encoder,
            model,
            device,
        }
    }

    /// Load a model checkpoint and wrap it as an evaluator.
    pub fn from_checkpoint(
        encoder: Arc<dyn StateEncoder<G>>,
        model: M,
        path: impl Into<std::path::PathBuf>,
        device: B::Device,
    ) -> Self
    where
        M: burn::module::Module<B>,
    {
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model = model
            .load_file(path, &recorder, &device)
            .expect("failed to load nn checkpoint");
        Self::new(encoder, model, device)
    }
}

// SAFETY: NeuralEvaluator only performs read-only inference (forward pass)
// through shared references.  Burn's Param<Tensor> is !Sync due to cubecl
// stub primitives, but after model loading all parameters are fully
// initialized and never mutated during inference.
unsafe impl<G, B: Backend, M: Send> Sync for NeuralEvaluator<G, B, M> {}

impl<G, B, M> Evaluator<G> for NeuralEvaluator<G, B, M>
where
    G: Game,
    B: Backend,
    M: PolicyValueNet<B> + Send,
{
    fn evaluate(&self, state: &G, _rng: &mut fastrand::Rng) -> Evaluation {
        let sign = match state.status() {
            Status::Ongoing => state.current_sign(),
            Status::Terminal(reward) => return Evaluation::uniform(G::NUM_ACTIONS, reward),
        };

        let feature_size = self.encoder.feature_size();
        let mut features = Vec::with_capacity(feature_size);
        self.encoder.encode(state, &mut features);

        let input =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, feature_size]), &self.device);

        let out = self.model.forward(input);
        let policy_logits_tensor = out.policy_logits;

        let logits_data: Vec<f32> = policy_logits_tensor
            .into_data()
            .to_vec()
            .expect("policy tensor to_vec");

        // Apply softmax to WDL logits → probabilities
        let wdl_probs = burn::tensor::activation::softmax(out.value, 1);
        let wdl_data: Vec<f32> = wdl_probs.into_data().to_vec().expect("value tensor to_vec");

        // WDL from current player's perspective; sign-flip for P1 (swap W/L)
        let wdl_cp = [wdl_data[0], wdl_data[1], wdl_data[2]];
        let wdl = if sign > 0.0 { wdl_cp } else { flip_wdl(wdl_cp) };
        debug_assert!(wdl.iter().all(|&x| x >= 0.0 && x <= 1.0));
        debug_assert!((wdl.iter().sum::<f32>() - 1.0).abs() < 1e-4);

        Evaluation {
            policy_logits: logits_data,
            wdl,
        }
    }

    fn evaluate_batch(&self, states: &[&G], rng: &mut fastrand::Rng) -> Vec<Evaluation> {
        if states.len() <= 1 {
            return states.iter().map(|s| self.evaluate(s, rng)).collect();
        }

        let feature_size = self.encoder.feature_size();

        // Determine which states are terminal vs ongoing; collect signs for ongoing
        let mut signs = Vec::with_capacity(states.len());
        let mut nn_indices = Vec::new();
        let mut results: Vec<Option<Evaluation>> = (0..states.len()).map(|_| None).collect();

        for (i, state) in states.iter().enumerate() {
            match state.status() {
                Status::Terminal(reward) => {
                    results[i] = Some(Evaluation::uniform(G::NUM_ACTIONS, reward));
                }
                Status::Ongoing => {
                    signs.push(state.current_sign());
                    nn_indices.push(i);
                }
            }
        }

        if nn_indices.is_empty() {
            return results.into_iter().map(|r| r.unwrap()).collect();
        }

        // Encode ongoing states and run inference
        let n = nn_indices.len();
        let mut features = Vec::with_capacity(n * feature_size);
        let mut buf = Vec::with_capacity(feature_size);
        for &i in &nn_indices {
            buf.clear();
            self.encoder.encode(states[i], &mut buf);
            features.extend_from_slice(&buf);
        }

        let (all_logits, all_wdl) = self.infer_features(features, n, feature_size);

        // Split outputs per state and apply sign correction
        for (j, &i) in nn_indices.iter().enumerate() {
            let logits_start = j * G::NUM_ACTIONS;
            let logits = all_logits[logits_start..logits_start + G::NUM_ACTIONS].to_vec();
            let wdl_start = j * 3;
            let wdl_raw = &all_wdl[wdl_start..wdl_start + 3];
            // Sign-flip for P1 perspective (swap W/L when sign < 0)
            let wdl_cp = [wdl_raw[0], wdl_raw[1], wdl_raw[2]];
            let wdl = if signs[j] > 0.0 {
                wdl_cp
            } else {
                flip_wdl(wdl_cp)
            };
            debug_assert!(wdl.iter().all(|&x| x >= 0.0 && x <= 1.0));
            debug_assert!((wdl.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            results[i] = Some(Evaluation {
                policy_logits: logits,
                wdl,
            });
        }

        results.into_iter().map(|r| r.unwrap()).collect()
    }

    fn infer_features(
        &self,
        features: Vec<f32>,
        batch_size: usize,
        feature_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(features, [batch_size, feature_size]),
            &self.device,
        );

        let out = self.model.forward(input);
        let policy_tensor = out.policy_logits;
        // Apply softmax to WDL logits → probabilities
        let wdl_tensor = burn::tensor::activation::softmax(out.value, 1);

        let logits: Vec<f32> = policy_tensor
            .into_data()
            .to_vec()
            .expect("policy tensor to_vec");
        let wdl: Vec<f32> = wdl_tensor
            .into_data()
            .to_vec()
            .expect("value tensor to_vec");

        debug_assert_eq!(wdl.len(), batch_size * 3);

        (logits, wdl)
    }
}
