use std::f32::consts::PI;

/// Sample Dirichlet noise: `n` samples from Gamma(alpha, 1), normalized.
pub fn sample(alpha: f32, n: usize, rng: &mut fastrand::Rng) -> Vec<f32> {
    let mut noise: Vec<f32> = (0..n).map(|_| sample_gamma(alpha, rng)).collect();
    let sum: f32 = noise.iter().sum();
    if sum > 0.0 {
        noise.iter_mut().for_each(|x| *x /= sum);
    }
    noise
}

/// Sample from Gamma(alpha, 1) using Marsaglia-Tsang with Box-Muller normals.
fn sample_gamma(alpha: f32, rng: &mut fastrand::Rng) -> f32 {
    if alpha < 1.0 {
        // Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
        return sample_gamma(alpha + 1.0, rng) * rng.f32().powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.f32();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Box-Muller normal sample.
fn sample_normal(rng: &mut fastrand::Rng) -> f32 {
    let u1 = 1.0 - rng.f32(); // avoid ln(0)
    let u2 = rng.f32();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
