use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use tokio::sync::mpsc;

/// A pre-encoded inference request sent from an async game task to the batcher.
/// Game-independent: carries raw features, not game states.
/// May contain multiple samples (a leaf batch) in a single request.
pub(super) struct InferRequest {
    pub flat_features: Vec<f32>,
    pub batch_size: usize,
    pub response_tx: tokio::sync::oneshot::Sender<InferResponse>,
}

/// Raw inference result returned from the batcher to a game task.
pub(super) struct InferResponse {
    /// Flat policy logits for all samples in the request.
    pub flat_policy_logits: Vec<f32>,
    /// One value per sample.
    pub values: Vec<f32>,
}

/// Sender half of the bounded inference request channel.
pub(super) type InferSender = mpsc::Sender<InferRequest>;

/// A batch of inference requests ready for GPU execution.
pub(super) struct InferBatch {
    flat_features: Vec<f32>,
    feature_size: usize,
    /// One entry per original request: (oneshot sender, number of samples in that request).
    responses: Vec<(tokio::sync::oneshot::Sender<InferResponse>, usize)>,
}

/// Shared atomic counters updated by the batcher, read by tasks for live stats.
pub(super) struct BatcherStats {
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
pub(super) fn batcher_loop(
    request_rx: &mut mpsc::Receiver<InferRequest>,
    work_tx: &crossbeam_channel::Sender<InferBatch>,
    max_batch_size: usize,
    stats: &BatcherStats,
) {
    let mut batch_requests: Vec<InferRequest> = Vec::with_capacity(max_batch_size);

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
        while total_samples < max_batch_size {
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

/// GPU worker loop: receives batches from the shared crossbeam queue,
/// runs forward passes, and sends results back via oneshot channels.
///
/// Each worker owns its own evaluator clone and runs on a dedicated thread.
/// Work-stealing is natural: the fastest GPU takes the next batch.
pub(super) fn gpu_worker_loop(
    batches: crossbeam_channel::Receiver<InferBatch>,
    infer_fn: impl Fn(Vec<f32>, usize, usize) -> (Vec<f32>, Vec<f32>),
    num_actions: usize,
) {
    while let Ok(batch) = batches.recv() {
        let total_samples: usize = batch.responses.iter().map(|(_, n)| n).sum();
        let (flat_logits, flat_values) =
            infer_fn(batch.flat_features, total_samples, batch.feature_size);

        let mut offset = 0;
        for (response_tx, count) in batch.responses {
            let logits = flat_logits[offset * num_actions..(offset + count) * num_actions].to_vec();
            let values = flat_values[offset..offset + count].to_vec();
            let _ = response_tx.send(InferResponse {
                flat_policy_logits: logits,
                values,
            });
            offset += count;
        }
    }
}
