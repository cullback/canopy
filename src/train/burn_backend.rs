//! Burn backend: `TrainableModel` implementation, loss computation, and device selection.
//!
//! Implements `TrainableModel` for any Burn model that satisfies `PolicyValueNet`.
//! Handles backend selection (CUDA > WGPU > ndarray), optimizer setup (AdamW),
//! train/val splitting, gradient steps with clipping, and checkpoint
//! serialization via Burn's recorder.

use std::path::Path;
use std::sync::Arc;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::activation::log_softmax;

use tracing_indicatif::span_ext::IndicatifSpanExt;

use crate::game::Game;
use crate::nn::{ForwardOutput, NeuralEvaluator, PolicyValueNet, StateEncoder};

use super::{Sample, TrainMetrics, TrainStepConfig, TrainableModel};

// Backend selection: cuda > wgpu > ndarray

#[cfg(feature = "cuda")]
use burn::backend::{Autodiff, Cuda};
#[cfg(feature = "cuda")]
pub type TrainBackend = Autodiff<Cuda>;
#[cfg(feature = "cuda")]
pub type InferBackend = Cuda;
#[cfg(feature = "cuda")]
pub type Device = burn::backend::cuda::CudaDevice;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use burn::backend::{Autodiff, Wgpu};
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type TrainBackend = Autodiff<Wgpu>;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type InferBackend = Wgpu;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type Device = burn::backend::wgpu::WgpuDevice;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
use burn::backend::{Autodiff, NdArray};
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type TrainBackend = Autodiff<NdArray>;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type InferBackend = NdArray;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type Device = burn::backend::ndarray::NdArrayDevice;

pub fn default_device() -> Device {
    Device::default()
}

/// Return the device for the `index`-th inference worker.
///
/// CUDA: maps to the GPU with that ordinal. Other backends: always the default device.
pub fn inference_device(index: usize) -> Device {
    #[cfg(feature = "cuda")]
    {
        burn::backend::cuda::CudaDevice::new(index)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = index;
        Device::default()
    }
}

/// All batch tensors needed for loss computation.
struct BatchTensors<B: Backend> {
    features: Tensor<B, 2>,
    policy_targets: Tensor<B, 2>,
    value_targets: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    num_full: usize,
    /// Auxiliary value targets [batch, num_aux]. None if no aux heads.
    aux_value_targets: Option<Tensor<B, 2>>,
}

/// Prepare batch tensors from a slice of samples.
fn prepare_batch<B: Backend>(
    samples: &[&Sample],
    cfg: &TrainStepConfig,
    feature_size: usize,
    num_actions: usize,
    device: &B::Device,
) -> BatchTensors<B> {
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

    // WDL value targets: blend one-hot z_wdl with search-refined q_wdl
    let value_targets: Vec<f32> = samples
        .iter()
        .flat_map(|s| {
            let z_wdl = if s.z > 0.0 {
                [1.0, 0.0, 0.0]
            } else if s.z < 0.0 {
                [0.0, 0.0, 1.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let alpha = cfg.q_weight;
            [
                (1.0 - alpha) * z_wdl[0] + alpha * s.q_wdl[0],
                (1.0 - alpha) * z_wdl[1] + alpha * s.q_wdl[1],
                (1.0 - alpha) * z_wdl[2] + alpha * s.q_wdl[2],
            ]
        })
        .collect();
    let value_tensor = Tensor::<B, 2>::from_data(TensorData::new(value_targets, [bs, 3]), device);

    let mask_data: Vec<f32> = samples
        .iter()
        .map(|s| if s.full_search { 1.0 } else { 0.0 })
        .collect();
    let mask_tensor = Tensor::<B, 2>::from_data(TensorData::new(mask_data, [bs, 1]), device);

    let num_full = samples.iter().filter(|s| s.full_search).count();

    // Auxiliary value targets
    let aux_value_targets = if cfg.num_aux_targets > 0 {
        let aux_data: Vec<f32> = samples
            .iter()
            .flat_map(|s| s.aux_targets.iter().copied())
            .collect();
        Some(Tensor::<B, 2>::from_data(
            TensorData::new(aux_data, [bs, cfg.num_aux_targets]),
            device,
        ))
    } else {
        None
    };

    BatchTensors {
        features: features_tensor,
        policy_targets: policy_tensor,
        value_targets: value_tensor,
        mask: mask_tensor,
        num_full,
        aux_value_targets,
    }
}

/// Losses returned from a single batch forward pass.
struct BatchLosses<B: Backend> {
    total: Tensor<B, 1>,
    policy: f32,
    wdl: f32,
    aux_value: f32,
    /// Per-horizon auxiliary value MSE (empty if no aux heads).
    aux_value_per_horizon: Vec<f32>,
}

/// Compute all losses from model output and batch tensors.
fn compute_batch_losses<B: Backend>(
    output: ForwardOutput<B>,
    batch: &BatchTensors<B>,
    cfg: &TrainStepConfig,
) -> BatchLosses<B> {
    let policy_logits = output.policy_logits;
    let value_pred = output.value;

    // Per-sample cross-entropy, masked by full_search (playout cap)
    let log_probs = log_softmax(policy_logits, 1);
    let per_sample_ce = batch.policy_targets.clone().mul(log_probs).sum_dim(1).neg(); // [batch, 1]
    let denom = if batch.num_full > 0 {
        batch.num_full as f32
    } else {
        1.0
    };

    let policy_loss = per_sample_ce
        .mul(batch.mask.clone())
        .sum()
        .div_scalar(denom);

    // Value loss: cross-entropy with WDL target
    let log_wdl = log_softmax(value_pred, 1); // [batch, 3]
    let per_sample_ce_value = batch.value_targets.clone().mul(log_wdl).sum_dim(1).neg(); // [batch, 1]
    let wdl_loss = per_sample_ce_value.mean();

    let pl = policy_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
    let wl = wdl_loss.clone().into_data().to_vec::<f32>().unwrap()[0];

    let mut total_loss = policy_loss.add(wdl_loss);

    // Auxiliary value loss
    let (aux_vl, aux_vl_per_horizon) = if let (Some(aux_targets), Some(aux_preds)) =
        (&batch.aux_value_targets, output.aux_values)
    {
        let diff = aux_preds.sub(aux_targets.clone()); // [batch, num_aux]
        let sq_diff = diff.powf_scalar(2.0); // [batch, num_aux]
        let per_sample_aux_mse = sq_diff.clone().mean_dim(1); // [batch, 1]
        let aux_loss = per_sample_aux_mse.mean();
        let al = aux_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        total_loss = total_loss.add(aux_loss.mul_scalar(cfg.aux_value_weight));
        // Per-horizon MSE (mean over batch)
        let per_horizon = sq_diff.mean_dim(0); // [1, num_aux]
        let per_horizon_vec = per_horizon
            .reshape([cfg.num_aux_targets])
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        (al, per_horizon_vec)
    } else {
        (0.0, vec![])
    };

    BatchLosses {
        total: total_loss.reshape([1]),
        policy: pl,
        wdl: wl,
        aux_value: aux_vl,
        aux_value_per_horizon: aux_vl_per_horizon,
    }
}

/// Aggregated losses from one epoch of training or validation.
#[derive(Default)]
struct EpochLosses {
    policy: f32,
    wdl: f32,
    aux_value: f32,
    aux_value_per_horizon: Vec<f32>,
}

pub struct BurnTrainableModel<G, M>
where
    M: AutodiffModule<TrainBackend>,
{
    model: M,
    optimizer: OptimizerAdaptor<AdamW, M, TrainBackend>,
    optim_config: AdamWConfig,
    device: Device,
    model_init: Box<dyn Fn(&Device) -> M + Send>,
    _game: std::marker::PhantomData<G>,
}

impl<G, M> BurnTrainableModel<G, M>
where
    M: AutodiffModule<TrainBackend>,
{
    pub fn new(model_init: impl Fn(&Device) -> M + Send + 'static) -> Self {
        let device = default_device();
        let model = model_init(&device);
        let optim_config = AdamWConfig::new()
            .with_weight_decay(0.0004)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));
        let optimizer = optim_config.clone().init();
        Self {
            model,
            optimizer,
            optim_config,
            device,
            model_init: Box::new(model_init),
            _game: std::marker::PhantomData,
        }
    }
}

impl<G, M> BurnTrainableModel<G, M>
where
    G: Game,
    M: AutodiffModule<TrainBackend> + PolicyValueNet<TrainBackend>,
    M::InnerModule: PolicyValueNet<InferBackend>,
{
    fn train_epoch(
        &mut self,
        samples: &[&Sample],
        cfg: &TrainStepConfig,
        span: &tracing::Span,
    ) -> (EpochLosses, usize) {
        let feature_size = samples[0].features.len();
        let mut model = std::mem::replace(&mut self.model, (self.model_init)(&self.device));

        let mut indices: Vec<usize> = (0..samples.len()).collect();
        fastrand::shuffle(&mut indices);

        let mut total_policy_loss = 0.0f32;
        let mut total_wdl_loss = 0.0f32;

        let mut total_aux_value_loss = 0.0f32;
        let mut total_aux_per_horizon = vec![0.0f32; cfg.num_aux_targets];
        let mut num_batches = 0usize;

        for batch_start in (0..indices.len()).step_by(cfg.train_batch_size) {
            let batch_end = (batch_start + cfg.train_batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];
            if batch_indices.is_empty() {
                continue;
            }

            let batch_samples: Vec<&Sample> = batch_indices.iter().map(|&i| samples[i]).collect();
            let batch = prepare_batch::<TrainBackend>(
                &batch_samples,
                cfg,
                feature_size,
                G::NUM_ACTIONS,
                &self.device,
            );

            let output = model.forward(batch.features.clone());
            let losses = compute_batch_losses(output, &batch, cfg);

            total_policy_loss += losses.policy;
            total_wdl_loss += losses.wdl;

            total_aux_value_loss += losses.aux_value;
            for (acc, &v) in total_aux_per_horizon
                .iter_mut()
                .zip(losses.aux_value_per_horizon.iter())
            {
                *acc += v;
            }
            num_batches += 1;

            let grads = losses.total.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = self.optimizer.step(cfg.lr, model, grads);
            span.pb_inc(1);
        }

        self.model = model;

        let nb = num_batches.max(1) as f32;
        for v in &mut total_aux_per_horizon {
            *v /= nb;
        }
        (
            EpochLosses {
                policy: total_policy_loss / nb,
                wdl: total_wdl_loss / nb,
                aux_value: total_aux_value_loss / nb,
                aux_value_per_horizon: total_aux_per_horizon,
            },
            num_batches,
        )
    }

    fn validate(&self, samples: &[&Sample], cfg: &TrainStepConfig) -> EpochLosses {
        let feature_size = samples[0].features.len();
        let model = self.model.valid();

        let mut total_policy_loss = 0.0f32;
        let mut total_wdl_loss = 0.0f32;

        let mut total_aux_value_loss = 0.0f32;
        let mut total_aux_per_horizon = vec![0.0f32; cfg.num_aux_targets];
        let mut num_batches = 0usize;

        for batch_start in (0..samples.len()).step_by(cfg.train_batch_size) {
            let batch_end = (batch_start + cfg.train_batch_size).min(samples.len());
            let batch_samples = &samples[batch_start..batch_end];
            if batch_samples.is_empty() {
                continue;
            }

            let batch = prepare_batch::<InferBackend>(
                batch_samples,
                cfg,
                feature_size,
                G::NUM_ACTIONS,
                &self.device,
            );

            let output = model.forward(batch.features.clone());
            let losses = compute_batch_losses(output, &batch, cfg);

            total_policy_loss += losses.policy;
            total_wdl_loss += losses.wdl;

            total_aux_value_loss += losses.aux_value;
            for (acc, &v) in total_aux_per_horizon
                .iter_mut()
                .zip(losses.aux_value_per_horizon.iter())
            {
                *acc += v;
            }
            num_batches += 1;
        }

        let nb = num_batches.max(1) as f32;
        for v in &mut total_aux_per_horizon {
            *v /= nb;
        }
        EpochLosses {
            policy: total_policy_loss / nb,
            wdl: total_wdl_loss / nb,
            aux_value: total_aux_value_loss / nb,
            aux_value_per_horizon: total_aux_per_horizon,
        }
    }
}

impl<G, M> TrainableModel<G> for BurnTrainableModel<G, M>
where
    G: Game + 'static,
    M: AutodiffModule<TrainBackend> + PolicyValueNet<TrainBackend> + Send + 'static,
    M::InnerModule: PolicyValueNet<InferBackend> + Send,
{
    fn evaluator(
        &self,
        encoder: Arc<dyn StateEncoder<G>>,
    ) -> Arc<dyn crate::eval::Evaluator<G> + Sync> {
        let infer_model = self.model.valid();
        Arc::new(NeuralEvaluator::new(
            encoder,
            infer_model,
            self.device.clone(),
        ))
    }

    fn evaluators(
        &self,
        encoder: Arc<dyn StateEncoder<G>>,
        count: usize,
    ) -> Vec<Arc<dyn crate::eval::Evaluator<G> + Sync>> {
        (0..count)
            .map(|i| {
                let device = inference_device(i);
                let infer_model = self.model.valid().to_device(&device);
                Arc::new(NeuralEvaluator::new(encoder.clone(), infer_model, device))
                    as Arc<dyn crate::eval::Evaluator<G> + Sync>
            })
            .collect()
    }

    fn train_step(&mut self, samples: &[&Sample], cfg: &TrainStepConfig) -> TrainMetrics {
        let val_split = (samples.len() * 4 / 5)
            .max(1)
            .min(samples.len().saturating_sub(1));
        let (train_samples, val_samples) = samples.split_at(val_split);

        let batches_per_epoch =
            (train_samples.len() + cfg.train_batch_size - 1) / cfg.train_batch_size;
        let total_batches = batches_per_epoch * cfg.epochs;

        let span = tracing::info_span!("training");
        span.pb_set_style(
            &indicatif::ProgressStyle::with_template(
                "{bar:40.cyan/dim} {pos}/{len} {per_sec}  {msg}  [{elapsed} < {eta}]",
            )
            .unwrap(),
        );
        span.pb_set_length(total_batches as u64);
        span.pb_set_message("training");
        span.pb_start();

        let mut final_train = EpochLosses::default();
        let mut final_val = EpochLosses::default();
        let mut total_gradient_steps = 0usize;

        for epoch in 0..cfg.epochs {
            let (train_losses, steps) = self.train_epoch(train_samples, cfg, &span);
            total_gradient_steps += steps;

            let val_losses = self.validate(val_samples, cfg);

            span.pb_set_message(&format!(
                "epoch {}/{} p={:.4} w={:.4} val_p={:.4} val_w={:.4}",
                epoch + 1,
                cfg.epochs,
                train_losses.policy,
                train_losses.wdl,
                val_losses.policy,
                val_losses.wdl
            ));

            final_train = train_losses;
            final_val = val_losses;
        }

        span.pb_set_finish_message(&format!(
            "p={:.4} w={:.4} val_p={:.4} val_w={:.4}",
            final_train.policy, final_train.wdl, final_val.policy, final_val.wdl,
        ));
        drop(span);

        TrainMetrics {
            loss_policy_train: final_train.policy,
            loss_wdl_train: final_train.wdl,
            loss_policy_val: final_val.policy,
            loss_wdl_val: final_val.wdl,
            loss_aux_value_train: final_train.aux_value,
            loss_aux_value_val: final_val.aux_value,
            aux_value_losses_train: final_train.aux_value_per_horizon,
            aux_value_losses_val: final_val.aux_value_per_horizon,
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
        self.load_weights(dir, iteration);

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let optim_path = dir.join(format!("optim_iter_{iteration}"));
        match recorder.load(optim_path, &self.device) {
            Ok(optim_record) => {
                let optimizer =
                    std::mem::replace(&mut self.optimizer, self.optim_config.clone().init());
                self.optimizer = optimizer.load_record(optim_record);
                tracing::info!("restored optimizer state");
            }
            Err(_) => {
                tracing::warn!("no optimizer state found, starting fresh optimizer");
            }
        }
    }

    fn load_weights(&mut self, dir: &Path, iteration: usize) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let path = dir.join(format!("model_iter_{iteration}"));
        self.model = (self.model_init)(&self.device)
            .load_file(path.to_str().unwrap(), &recorder, &self.device)
            .expect("failed to load checkpoint");
    }
}
