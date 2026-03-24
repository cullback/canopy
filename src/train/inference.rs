//! Batched GPU inference with pause/resume and evaluator hot-swap.
//!
//! Async game tasks send pre-encoded `InferRequest`s to a batcher thread,
//! which packs them into `InferBatch`es and dispatches to GPU worker threads
//! via a crossbeam SPMC queue. Each `InferenceServer` manages one GPU with
//! its own batcher + worker pair, supporting pause (for training on GPU 0)
//! and live evaluator replacement. `InferencePipeline` is a simpler
//! multi-GPU variant without pause/swap for tournament use.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use tokio::sync::mpsc;

use crate::eval::Evaluator;
use crate::game::Game;

/// A pre-encoded inference request sent from an async game task to the batcher.
/// Game-independent: carries raw features, not game states.
///
/// A single request may contain multiple samples (e.g. `leaf_batch_size`
/// leaves from one MCTS step), packed contiguously in `flat_features`.
pub struct InferRequest {
    /// `[batch_size * feature_size]` floats — one or more encoded states
    /// packed contiguously.
    pub flat_features: Vec<f32>,
    /// Number of samples in `flat_features`. The batcher derives
    /// `feature_size = flat_features.len() / batch_size` and uses this
    /// count to aggregate samples across requests and slice responses.
    pub batch_size: usize,
    pub response_tx: tokio::sync::oneshot::Sender<InferResponse>,
}

/// Raw inference result returned from the batcher to a game task.
pub struct InferResponse {
    /// Flat policy logits for all samples in the request.
    pub flat_policy_logits: Vec<f32>,
    /// WDL probabilities, 3 per sample (current player's perspective).
    pub flat_wdl: Vec<f32>,
}

/// Sender half of the bounded inference request channel.
pub type InferSender = mpsc::Sender<InferRequest>;

/// A batch of inference requests ready for GPU execution.
pub struct InferBatch {
    flat_features: Vec<f32>,
    feature_size: usize,
    /// One entry per original request: (oneshot sender, number of samples in that request).
    responses: Vec<(tokio::sync::oneshot::Sender<InferResponse>, usize)>,
}

/// Shared atomic counters updated by the batcher, read by tasks for live stats.
pub struct BatcherStats {
    pub batches: AtomicU64,
    pub evals: AtomicU64,
}

impl BatcherStats {
    pub fn new() -> Self {
        Self {
            batches: AtomicU64::new(0),
            evals: AtomicU64::new(0),
        }
    }

    pub fn avg_batch_size(&self) -> f64 {
        let b = self.batches.load(Relaxed);
        let e = self.evals.load(Relaxed);
        if b == 0 { 0.0 } else { e as f64 / b as f64 }
    }
}

/// Batcher thread: collects pre-encoded feature vectors from actors,
/// forms batches, and dispatches them to GPU workers via a crossbeam SPMC queue.
///
/// Pure routing — no model or GPU knowledge.
pub fn batcher_loop(
    request_rx: &mut mpsc::Receiver<InferRequest>,
    work_tx: &crossbeam_channel::Sender<InferBatch>,
    inference_batch_size: usize,
    stats: &BatcherStats,
) {
    let mut batch_requests: Vec<InferRequest> = Vec::with_capacity(inference_batch_size);

    loop {
        // Block for the first request (None = all senders dropped)
        let first = match request_rx.blocking_recv() {
            Some(req) => req,
            None => break,
        };

        let feature_size = first.flat_features.len() / first.batch_size;
        batch_requests.push(first);

        // Drain additional requests without blocking
        let mut total_samples: usize = batch_requests[0].batch_size;
        while total_samples < inference_batch_size {
            match request_rx.try_recv() {
                Ok(req) => {
                    total_samples += req.batch_size;
                    batch_requests.push(req);
                }
                Err(_) => break,
            }
        }

        stats.evals.fetch_add(total_samples as u64, Relaxed);
        stats.batches.fetch_add(1, Relaxed);

        // Collect flat features and response handles
        let mut flat_features = Vec::with_capacity(total_samples * feature_size);
        let mut responses = Vec::with_capacity(batch_requests.len());
        for req in batch_requests.drain(..) {
            flat_features.extend_from_slice(&req.flat_features);
            responses.push((req.response_tx, req.batch_size));
        }

        // Send batch to GPU worker pool (blocks if all workers busy — backpressure)
        let batch = InferBatch {
            flat_features,
            feature_size,
            responses,
        };
        if work_tx.send(batch).is_err() {
            // All workers gone — shutting down
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Pause mechanism
// ---------------------------------------------------------------------------

/// Pause/resume control for GPU 0 during training.
/// Uses Mutex+Condvar — uncontended lock when not paused (nanoseconds).
pub struct PauseControl {
    inner: std::sync::Mutex<bool>,
    condvar: std::sync::Condvar,
}

impl PauseControl {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(false),
            condvar: std::sync::Condvar::new(),
        }
    }

    pub fn pause(&self) {
        *self.inner.lock().unwrap() = true;
    }

    pub fn resume(&self) {
        *self.inner.lock().unwrap() = false;
        self.condvar.notify_all();
    }

    pub fn wait_if_paused(&self) {
        let guard = self.inner.lock().unwrap();
        let _guard = self.condvar.wait_while(guard, |paused| *paused).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Hot-swappable evaluator
// ---------------------------------------------------------------------------

pub type InferFn = dyn Fn(Vec<f32>, usize, usize) -> (Vec<f32>, Vec<f32>) + Send + Sync;

pub struct SwappableInferFn {
    inner: std::sync::Mutex<Arc<InferFn>>,
}

impl SwappableInferFn {
    pub fn new(f: Arc<InferFn>) -> Self {
        Self {
            inner: std::sync::Mutex::new(f),
        }
    }

    /// Clone the current Arc (nanoseconds, uncontended).
    pub fn load(&self) -> Arc<InferFn> {
        self.inner.lock().unwrap().clone()
    }

    /// Swap in a new evaluator function.
    pub fn store(&self, new: Arc<InferFn>) {
        *self.inner.lock().unwrap() = new;
    }
}

// ---------------------------------------------------------------------------
// GPU worker loop (with pause + hot-swap)
// ---------------------------------------------------------------------------

fn process_batch(infer_fn: &InferFn, batch: InferBatch, num_actions: usize) {
    let total_samples: usize = batch.responses.iter().map(|(_, n)| n).sum();
    let (flat_logits, flat_wdl) = infer_fn(batch.flat_features, total_samples, batch.feature_size);
    debug_assert_eq!(flat_wdl.len(), total_samples * 3);

    let mut offset = 0;
    for (response_tx, count) in batch.responses {
        let logits = flat_logits[offset * num_actions..(offset + count) * num_actions].to_vec();
        let wdl = flat_wdl[offset * 3..(offset + count) * 3].to_vec();
        let _ = response_tx.send(InferResponse {
            flat_policy_logits: logits,
            flat_wdl: wdl,
        });
        offset += count;
    }
}

/// GPU worker loop: receives batches, runs forward passes, sends results.
///
/// Supports optional pause control (GPU 0) and hot-swappable evaluator.
pub fn gpu_worker_loop(
    batches: crossbeam_channel::Receiver<InferBatch>,
    infer_fn: Arc<SwappableInferFn>,
    num_actions: usize,
    pause: Option<Arc<PauseControl>>,
) {
    loop {
        if let Some(ref p) = pause {
            p.wait_if_paused();
        }
        let batch = match batches.recv() {
            Ok(b) => b,
            Err(_) => break,
        };
        let f = infer_fn.load();
        process_batch(&*f, batch, num_actions);
    }
}

// ---------------------------------------------------------------------------
// Persistent InferenceServer (one per GPU)
// ---------------------------------------------------------------------------

/// Manages batcher + GPU worker threads for a single GPU.
///
/// Persistent across iterations — supports pause/resume and evaluator hot-swap.
pub struct InferenceServer {
    request_tx: mpsc::Sender<InferRequest>,
    batcher_handle: std::thread::JoinHandle<()>,
    worker_handle: std::thread::JoinHandle<()>,
    infer_fn: Arc<SwappableInferFn>,
    stats: Arc<BatcherStats>,
}

impl InferenceServer {
    /// Start a persistent inference server for one GPU.
    pub fn start<G: Game + 'static>(
        eval: Arc<dyn Evaluator<G> + Sync>,
        batch_size: usize,
        pause: Option<Arc<PauseControl>>,
    ) -> Self {
        let num_actions = G::NUM_ACTIONS;
        let (request_tx, mut request_rx) = mpsc::channel(2 * batch_size);
        let (work_tx, work_rx) = crossbeam_channel::bounded(2);
        let stats = Arc::new(BatcherStats::new());
        let stats_ref = stats.clone();

        let batcher_handle = std::thread::spawn(move || {
            batcher_loop(&mut request_rx, &work_tx, batch_size, &stats_ref);
        });

        let infer_fn = Arc::new(SwappableInferFn::new(make_infer_fn::<G>(eval)));
        let infer_fn_ref = infer_fn.clone();

        let worker_handle = std::thread::spawn(move || {
            gpu_worker_loop(work_rx, infer_fn_ref, num_actions, pause);
        });

        Self {
            request_tx,
            batcher_handle,
            worker_handle,
            infer_fn,
            stats,
        }
    }

    /// Clone the request sender (for worker construction).
    pub fn sender(&self) -> mpsc::Sender<InferRequest> {
        self.request_tx.clone()
    }

    pub fn stats(&self) -> &Arc<BatcherStats> {
        &self.stats
    }

    /// Hot-swap the evaluator function (takes effect on next batch).
    pub fn swap_infer_fn(&self, new: Arc<InferFn>) {
        self.infer_fn.store(new);
    }

    /// Shut down the server: drop sender, join threads.
    pub fn shutdown(self) {
        drop(self.request_tx);
        self.batcher_handle.join().unwrap();
        self.worker_handle.join().unwrap();
    }
}

/// Create an InferFn from an Evaluator.
pub fn make_infer_fn<G: Game + 'static>(eval: Arc<dyn Evaluator<G> + Sync>) -> Arc<InferFn> {
    Arc::new(move |features, batch_size, feature_size| {
        eval.infer_features(features, batch_size, feature_size)
    })
}

// ---------------------------------------------------------------------------
// InferencePipeline — simple multi-GPU pipeline for tournament use
// ---------------------------------------------------------------------------

/// Simple inference pipeline for non-training contexts (tournaments, serve).
///
/// Unlike [`InferenceServer`], this supports multiple GPU workers sharing a
/// single batcher via crossbeam SPMC, but has no pause/resume or hot-swap.
pub struct InferencePipeline {
    request_tx: mpsc::Sender<InferRequest>,
    batcher_handle: std::thread::JoinHandle<()>,
    worker_handles: Vec<std::thread::JoinHandle<()>>,
}

impl InferencePipeline {
    pub fn start<G: Game + 'static>(
        evaluators: Vec<Arc<dyn Evaluator<G> + Sync>>,
        inference_batch_size: usize,
    ) -> Self {
        let num_actions = G::NUM_ACTIONS;
        let (request_tx, mut request_rx) = mpsc::channel(2 * inference_batch_size);
        let (work_tx, work_rx) = crossbeam_channel::bounded(2);
        let stats = Arc::new(BatcherStats::new());
        let stats_ref = stats.clone();

        let batcher_handle = std::thread::spawn(move || {
            batcher_loop(&mut request_rx, &work_tx, inference_batch_size, &stats_ref);
        });

        let mut worker_handles = Vec::new();
        for evaluator in evaluators {
            let rx = work_rx.clone();
            let infer_fn = Arc::new(SwappableInferFn::new(make_infer_fn::<G>(evaluator)));
            worker_handles.push(std::thread::spawn(move || {
                gpu_worker_loop(rx, infer_fn, num_actions, None);
            }));
        }
        drop(work_rx);

        Self {
            request_tx,
            batcher_handle,
            worker_handles,
        }
    }

    pub fn sender(&self) -> mpsc::Sender<InferRequest> {
        self.request_tx.clone()
    }

    pub fn join(self) {
        drop(self.request_tx);
        self.batcher_handle.join().unwrap();
        for h in self.worker_handles {
            h.join().unwrap();
        }
    }
}
