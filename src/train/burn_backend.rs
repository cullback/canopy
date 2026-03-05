use std::marker::PhantomData;
use std::path::Path;

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::activation::log_softmax;

use crate::game::Game;
use crate::nn::{NeuralEvaluator, PolicyValueNet, StateEncoder};

use super::{Sample, TrainMetrics, TrainStepConfig, TrainableModel};

// Backend selection: cuda > wgpu > ndarray

#[cfg(feature = "cuda")]
use burn::backend::{Autodiff, Cuda};
#[cfg(feature = "cuda")]
type TrainBackend = Autodiff<Cuda>;
#[cfg(feature = "cuda")]
pub type InferBackend = Cuda;
#[cfg(feature = "cuda")]
pub type Device = burn::backend::cuda::CudaDevice;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use burn::backend::{Autodiff, Wgpu};
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type TrainBackend = Autodiff<Wgpu>;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type InferBackend = Wgpu;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type Device = burn::backend::wgpu::WgpuDevice;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
use burn::backend::{Autodiff, NdArray};
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
type TrainBackend = Autodiff<NdArray>;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type InferBackend = NdArray;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type Device = burn::backend::ndarray::NdArrayDevice;

pub fn default_device() -> Device {
    Device::default()
}

/// Prepare batch tensors from a slice of samples.
/// Returns (features, policy_targets, value_targets, mask, num_full) tensors.
fn prepare_batch<B: Backend>(
    samples: &[&Sample],
    alpha: f32,
    feature_size: usize,
    num_actions: usize,
    device: &B::Device,
) -> (
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    usize,
) {
    let bs = samples.len();

    let features: Vec<f32> = samples
        .iter()
        .flat_map(|s| s.features.iter().copied())
        .collect();
    let features_tensor =
        Tensor::<B, 2>::from_data(TensorData::new(features, [bs, feature_size]), device);

    let policy_targets: Vec<f32> = samples
        .iter()
        .flat_map(|s| s.policy_target.iter().copied())
        .collect();
    let policy_tensor =
        Tensor::<B, 2>::from_data(TensorData::new(policy_targets, [bs, num_actions]), device);

    let value_targets: Vec<f32> = samples
        .iter()
        .map(|s| (1.0 - alpha) * s.z + alpha * s.q)
        .collect();
    let value_tensor = Tensor::<B, 2>::from_data(TensorData::new(value_targets, [bs, 1]), device);

    let mask_data: Vec<f32> = samples
        .iter()
        .map(|s| if s.full_search { 1.0 } else { 0.0 })
        .collect();
    let mask_tensor = Tensor::<B, 2>::from_data(TensorData::new(mask_data, [bs, 1]), device);

    let num_full = samples.iter().filter(|s| s.full_search).count();

    (
        features_tensor,
        policy_tensor,
        value_tensor,
        mask_tensor,
        num_full,
    )
}

/// Compute policy and value losses from model output and batch tensors.
fn compute_batch_losses<B: Backend>(
    policy_logits: Tensor<B, 2>,
    value_pred: Tensor<B, 2>,
    policy_tensor: Tensor<B, 2>,
    value_tensor: Tensor<B, 2>,
    mask_tensor: Tensor<B, 2>,
    num_full: usize,
) -> (Tensor<B, 1>, f32, f32) {
    // Per-sample cross-entropy, masked by full_search (playout cap)
    let log_probs = log_softmax(policy_logits, 1);
    let per_sample_ce = policy_tensor.mul(log_probs).sum_dim(1).neg();
    let denom = if num_full > 0 { num_full as f32 } else { 1.0 };
    let policy_loss = per_sample_ce.mul(mask_tensor).sum().div_scalar(denom);

    let value_diff = value_pred.sub(value_tensor);
    let value_loss = value_diff.powf_scalar(2.0).mean();

    let pl = policy_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
    let vl = value_loss.clone().into_data().to_vec::<f32>().unwrap()[0];

    let loss = policy_loss.add(value_loss).reshape([1]);
    (loss, pl, vl)
}

pub struct BurnTrainableModel<G, E, M>
where
    M: AutodiffModule<TrainBackend>,
{
    model: M,
    optimizer: OptimizerAdaptor<AdamW, M, TrainBackend>,
    device: Device,
    model_init: Box<dyn Fn(&Device) -> M + Send>,
    _marker: PhantomData<fn() -> (G, E)>,
}

impl<G, E, M> BurnTrainableModel<G, E, M>
where
    M: AutodiffModule<TrainBackend>,
{
    pub fn new(model_init: impl Fn(&Device) -> M + Send + 'static, device: &Device) -> Self {
        let model = model_init(device);
        let optimizer = AdamWConfig::new().with_weight_decay(0.0004).init();
        Self {
            model,
            optimizer,
            device: device.clone(),
            model_init: Box::new(model_init),
            _marker: PhantomData,
        }
    }
}

impl<G, E, M> BurnTrainableModel<G, E, M>
where
    G: Game,
    E: StateEncoder<G>,
    M: AutodiffModule<TrainBackend> + PolicyValueNet<TrainBackend>,
    M::InnerModule: PolicyValueNet<InferBackend>,
{
    fn train_epoch(&mut self, samples: &[&Sample], cfg: &TrainStepConfig) -> (f32, f32, usize) {
        let mut model = std::mem::replace(&mut self.model, (self.model_init)(&self.device));

        let mut indices: Vec<usize> = (0..samples.len()).collect();
        fastrand::shuffle(&mut indices);

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut num_batches = 0usize;

        for batch_start in (0..indices.len()).step_by(cfg.batch_size) {
            let batch_end = (batch_start + cfg.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];
            if batch_indices.is_empty() {
                continue;
            }

            let batch_samples: Vec<&Sample> = batch_indices.iter().map(|&i| samples[i]).collect();
            let (features_tensor, policy_tensor, value_tensor, mask_tensor, num_full) =
                prepare_batch::<TrainBackend>(
                    &batch_samples,
                    cfg.alpha,
                    E::FEATURE_SIZE,
                    G::NUM_ACTIONS,
                    &self.device,
                );

            let (policy_logits, value_pred) = model.forward(features_tensor);
            let (loss, pl, vl) = compute_batch_losses(
                policy_logits,
                value_pred,
                policy_tensor,
                value_tensor,
                mask_tensor,
                num_full,
            );

            total_policy_loss += pl;
            total_value_loss += vl;
            num_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = self.optimizer.step(cfg.lr, model, grads);
        }

        self.model = model;

        let avg_policy = if num_batches > 0 {
            total_policy_loss / num_batches as f32
        } else {
            0.0
        };
        let avg_value = if num_batches > 0 {
            total_value_loss / num_batches as f32
        } else {
            0.0
        };
        (avg_policy, avg_value, num_batches)
    }

    fn validate(&self, samples: &[&Sample], cfg: &TrainStepConfig) -> (f32, f32) {
        let model = self.model.valid();

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut num_batches = 0usize;

        for batch_start in (0..samples.len()).step_by(cfg.batch_size) {
            let batch_end = (batch_start + cfg.batch_size).min(samples.len());
            let batch = &samples[batch_start..batch_end];
            if batch.is_empty() {
                continue;
            }

            let (features_tensor, policy_tensor, value_tensor, mask_tensor, num_full) =
                prepare_batch::<InferBackend>(
                    batch,
                    cfg.alpha,
                    E::FEATURE_SIZE,
                    G::NUM_ACTIONS,
                    &self.device,
                );

            let (policy_logits, value_pred) = model.forward(features_tensor);
            let (_loss, pl, vl) = compute_batch_losses(
                policy_logits,
                value_pred,
                policy_tensor,
                value_tensor,
                mask_tensor,
                num_full,
            );

            total_policy_loss += pl;
            total_value_loss += vl;
            num_batches += 1;
        }

        let avg_policy = if num_batches > 0 {
            total_policy_loss / num_batches as f32
        } else {
            0.0
        };
        let avg_value = if num_batches > 0 {
            total_value_loss / num_batches as f32
        } else {
            0.0
        };
        (avg_policy, avg_value)
    }
}

impl<G, E, M> TrainableModel<G> for BurnTrainableModel<G, E, M>
where
    G: Game,
    E: StateEncoder<G> + Send,
    M: AutodiffModule<TrainBackend> + PolicyValueNet<TrainBackend> + Send + 'static,
    M::InnerModule: PolicyValueNet<InferBackend> + Send,
{
    type Encoder = E;
    type Evaluator = NeuralEvaluator<InferBackend, E, M::InnerModule>;

    fn evaluator(&self) -> Self::Evaluator {
        let infer_model = self.model.valid();
        NeuralEvaluator::new(infer_model, self.device.clone())
    }

    fn train_step(&mut self, samples: &[&Sample], cfg: &TrainStepConfig) -> TrainMetrics {
        let val_split = (samples.len() * 4 / 5)
            .max(1)
            .min(samples.len().saturating_sub(1));
        let (train_samples, val_samples) = samples.split_at(val_split);

        let mut final_train_ploss = 0.0f32;
        let mut final_train_vloss = 0.0f32;
        let mut final_val_ploss = 0.0f32;
        let mut final_val_vloss = 0.0f32;
        let mut total_gradient_steps = 0usize;

        for epoch in 0..cfg.epochs {
            let (ploss, vloss, steps) = self.train_epoch(train_samples, cfg);
            total_gradient_steps += steps;

            let (val_ploss, val_vloss) = self.validate(val_samples, cfg);

            final_train_ploss = ploss;
            final_train_vloss = vloss;
            final_val_ploss = val_ploss;
            final_val_vloss = val_vloss;

            eprintln!(
                "  epoch {}/{}: train(p={:.4} v={:.4}) val(p={:.4} v={:.4})",
                epoch + 1,
                cfg.epochs,
                ploss,
                vloss,
                val_ploss,
                val_vloss
            );
        }

        TrainMetrics {
            train_policy_loss: final_train_ploss,
            train_value_loss: final_train_vloss,
            val_policy_loss: final_val_ploss,
            val_value_loss: final_val_vloss,
            gradient_steps: total_gradient_steps,
        }
    }

    fn save(&self, dir: &Path, iteration: usize) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        let path = dir.join(format!("model_iter_{iteration}"));
        self.model
            .clone()
            .save_file(path.to_str().unwrap(), &recorder)
            .expect("failed to save checkpoint");

        let optim_path = dir.join(format!("optim_iter_{iteration}"));
        recorder
            .record(self.optimizer.to_record(), optim_path)
            .expect("failed to save optimizer state");
    }

    fn load(&mut self, dir: &Path, iteration: usize) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        let path = dir.join(format!("model_iter_{iteration}"));
        self.model = (self.model_init)(&self.device)
            .load_file(path.to_str().unwrap(), &recorder, &self.device)
            .expect("failed to load checkpoint");

        let optim_path = dir.join(format!("optim_iter_{iteration}"));
        match recorder.load(optim_path, &self.device) {
            Ok(optim_record) => {
                let optimizer = std::mem::replace(
                    &mut self.optimizer,
                    AdamWConfig::new().with_weight_decay(0.0004).init(),
                );
                self.optimizer = optimizer.load_record(optim_record);
                eprintln!("restored optimizer state");
            }
            Err(_) => {
                eprintln!("warning: no optimizer state found, starting fresh optimizer");
            }
        }
    }
}
