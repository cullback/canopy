use canopy2::nn::StateEncoder;

use crate::game::{Board, get_nibble};

pub struct Twenty48Encoder;

impl Twenty48Encoder {
    /// 16 cells x 16 possible exponent values = 256 features.
    pub const FEATURE_SIZE: usize = 256;
}

impl StateEncoder<Board> for Twenty48Encoder {
    fn feature_size(&self) -> usize {
        Self::FEATURE_SIZE
    }

    fn encode(&self, state: &Board, out: &mut Vec<f32>) {
        out.clear();
        out.resize(Self::FEATURE_SIZE, 0.0);
        for cell in 0..16u32 {
            let nibble = get_nibble(state.tiles, cell) as usize;
            out[cell as usize * 16 + nibble] = 1.0;
        }
        debug_assert_eq!(out.len(), self.feature_size());
    }
}
