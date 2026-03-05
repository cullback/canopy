mod resnet;
mod simple;

pub use resnet::CatanResModelConfig;
pub use simple::{CatanModel, CatanModelConfig};

use burn::nn::Linear;
use burn::prelude::*;

// Stream slices (must match encoder layout exactly)
const GLOBAL_START: usize = 0;
const GLOBAL_LEN: usize = 50;
const TILES_START: usize = 50;
const TILES_N: usize = 19;
const TILES_F: usize = 7;
const NODES_START: usize = 183;
const NODES_N: usize = 54;
const NODES_F: usize = 2;
const EDGES_START: usize = 291;
const EDGES_N: usize = 72;
const EDGES_F: usize = 2;
const PORTS_START: usize = 435;
const PORTS_N: usize = 9;
const PORTS_F: usize = 5;

const STREAM_DIM: usize = 64 + 32 + 32 + 16 + 16; // 160

fn relu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::relu(x)
}

/// Project a per-element stream: [bs, N*F] → Linear+ReLU → mean → [bs, out].
fn project_stream<B: Backend>(
    input: &Tensor<B, 2>,
    start: usize,
    n: usize,
    f: usize,
    linear: &Linear<B>,
) -> Tensor<B, 2> {
    let [bs, _] = input.dims();
    let slice = input.clone().narrow(1, start, n * f);
    // [bs, N*F] → [bs*N, F]
    let flat = slice.reshape([bs * n, f]);
    let projected = relu(linear.forward(flat));
    // [bs*N, out] → [bs, N, out]
    let out_dim = projected.dims()[1];
    let grouped = projected.reshape([bs, n, out_dim]);
    // mean over dim 1 → [bs, out]
    grouped.mean_dim(1).reshape([bs, out_dim])
}

/// Project the global stream: [bs, 50] → Linear+ReLU → [bs, 64].
fn project_global<B: Backend>(input: &Tensor<B, 2>, linear: &Linear<B>) -> Tensor<B, 2> {
    let slice = input.clone().narrow(1, GLOBAL_START, GLOBAL_LEN);
    relu(linear.forward(slice))
}

/// Run all stream projections and concatenate to [bs, 160].
fn project_streams<B: Backend>(
    input: &Tensor<B, 2>,
    global: &Linear<B>,
    tiles: &Linear<B>,
    nodes: &Linear<B>,
    edges: &Linear<B>,
    ports: &Linear<B>,
) -> Tensor<B, 2> {
    let g = project_global(input, global);
    let t = project_stream(input, TILES_START, TILES_N, TILES_F, tiles);
    let n = project_stream(input, NODES_START, NODES_N, NODES_F, nodes);
    let e = project_stream(input, EDGES_START, EDGES_N, EDGES_F, edges);
    let p = project_stream(input, PORTS_START, PORTS_N, PORTS_F, ports);
    Tensor::cat(vec![g, t, n, e, p], 1)
}
