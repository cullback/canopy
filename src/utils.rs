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
