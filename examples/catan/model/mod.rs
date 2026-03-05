mod resnet;
mod simple;

pub use resnet::CatanResModelConfig;
pub use simple::{CatanModel, CatanModelConfig};

use burn::nn::Linear;
use burn::prelude::*;

// Stream constants — fixed board topology sizes
const GLOBAL_START: usize = 0;
const GLOBAL_LEN: usize = 49;
const TILES_N: usize = 19;
const NODES_N: usize = 54;
const EDGES_N: usize = 72;
const PORTS_N: usize = 9;

// Projection output dimensions
const GLOBAL_OUT: usize = 64;
const TILES_OUT: usize = 32;
const NODES_OUT: usize = 32;
const EDGES_OUT: usize = 16;
const PORTS_OUT: usize = 16;

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

/// Project the global stream: [bs, GLOBAL_LEN] → Linear+ReLU → [bs, GLOBAL_OUT].
fn project_global<B: Backend>(input: &Tensor<B, 2>, linear: &Linear<B>) -> Tensor<B, 2> {
    let slice = input.clone().narrow(1, GLOBAL_START, GLOBAL_LEN);
    relu(linear.forward(slice))
}

/// Run all stream projections and concatenate.
///
/// Supports variable encoder layouts: tile and port streams are optional.
/// Stream feature dimensions are read from the Linear weight shapes so the
/// function adapts to any encoder without stored layout metadata.
fn project_streams<B: Backend>(
    input: &Tensor<B, 2>,
    proj_global: &Linear<B>,
    proj_tiles: &Option<Linear<B>>,
    proj_nodes: &Linear<B>,
    proj_edges: &Linear<B>,
    proj_ports: &Option<Linear<B>>,
) -> Tensor<B, 2> {
    let mut parts = vec![project_global(input, proj_global)];
    let mut offset = GLOBAL_LEN;

    // Tile stream (optional)
    if let Some(ref tiles) = *proj_tiles {
        let tiles_f = tiles.weight.dims()[1];
        parts.push(project_stream(input, offset, TILES_N, tiles_f, tiles));
        offset += TILES_N * tiles_f;
    }

    // Node stream (always present)
    let nodes_f = proj_nodes.weight.dims()[1];
    parts.push(project_stream(input, offset, NODES_N, nodes_f, proj_nodes));
    offset += NODES_N * nodes_f;

    // Edge stream (always present)
    let edges_f = proj_edges.weight.dims()[1];
    parts.push(project_stream(input, offset, EDGES_N, edges_f, proj_edges));
    offset += EDGES_N * edges_f;

    // Port stream (optional)
    if let Some(ref ports) = *proj_ports {
        let ports_f = ports.weight.dims()[1];
        parts.push(project_stream(input, offset, PORTS_N, ports_f, ports));
    }

    Tensor::cat(parts, 1)
}

/// Compute the concatenated stream dimension from config parameters.
fn stream_dim(tiles_f: usize, ports_f: usize) -> usize {
    let mut dim = GLOBAL_OUT + NODES_OUT + EDGES_OUT;
    if tiles_f > 0 {
        dim += TILES_OUT;
    }
    if ports_f > 0 {
        dim += PORTS_OUT;
    }
    dim
}
