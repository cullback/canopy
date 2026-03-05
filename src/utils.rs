use std::fmt;

/// Softmax over a subset of `logits` indicated by `indices`.
/// Returns a probability distribution of the same length as `indices`.
pub fn softmax_masked(logits: &[f32], indices: &[usize]) -> Vec<f32> {
    let max = indices
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = indices.iter().map(|&i| (logits[i] - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    probs
}

pub struct HumanDuration(pub std::time::Duration);

impl fmt::Display for HumanDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.as_secs();
        if secs >= 3600 {
            write!(f, "{}h{:02}m", secs / 3600, (secs % 3600) / 60)
        } else if secs >= 60 {
            write!(f, "{}m{:02}s", secs / 60, secs % 60)
        } else {
            write!(f, "{}s", secs)
        }
    }
}
