use canopy::nn::StateEncoder;

use crate::game::{Board, get_nibble};

pub struct Twenty48Encoder;

/// Number of one-hot channels: 0 = empty, 1–17 = tile exponent (2^k).
const NUM_CHANNELS: usize = 18;

impl Twenty48Encoder {
    /// 18 channels × 4 rows × 4 cols = 288 features (channel-major for Conv2d).
    pub const FEATURE_SIZE: usize = NUM_CHANNELS * 16;
}

impl StateEncoder<Board> for Twenty48Encoder {
    fn feature_size(&self) -> usize {
        Self::FEATURE_SIZE
    }

    fn encode(&self, state: &Board, out: &mut Vec<f32>) {
        out.clear();
        out.resize(Self::FEATURE_SIZE, 0.0);
        let tiles = state.tiles();
        for cell in 0..16u32 {
            let nibble = get_nibble(tiles, cell) as usize;
            let row = cell as usize / 4;
            let col = cell as usize % 4;
            // Channel-major: index = channel * 16 + row * 4 + col
            out[nibble * 16 + row * 4 + col] = 1.0;
        }
        debug_assert_eq!(out.len(), self.feature_size());
    }
}
