use canopy::nn::StateEncoder;

use crate::game::{K, Phase, TenureGame, optimal_value};

pub struct TenureEncoder;

impl TenureEncoder {
    pub const FEATURE_SIZE: usize = K + K + 3; // 23
}

impl StateEncoder<TenureGame> for TenureEncoder {
    fn feature_size(&self) -> usize {
        Self::FEATURE_SIZE
    }

    fn encode(&self, state: &TenureGame, out: &mut Vec<f32>) {
        out.clear();

        let total: u8 = state.board.iter().sum::<u8>() + state.partition.iter().sum::<u8>();
        let norm = if total > 0 { total as f32 } else { 1.0 };

        // Board levels normalized by total pieces (10 features)
        for i in 0..K {
            out.push(state.board[i] as f32 / norm);
        }

        // Partition levels normalized by total pieces (10 features)
        for i in 0..K {
            out.push(state.partition[i] as f32 / norm);
        }

        // Phase: 1.0 = attacker, 0.0 = defender
        out.push(match state.phase {
            Phase::Attacker => 1.0,
            Phase::Defender => 0.0,
        });

        // Score / initial_value
        if state.initial_value > 0.0 {
            out.push(state.score as f32 / state.initial_value);
        } else {
            out.push(0.0);
        }

        // Remaining optimal value / initial_value
        let remaining = optimal_value(&state.board) + optimal_value(&state.partition);
        if state.initial_value > 0.0 {
            out.push(remaining / state.initial_value);
        } else {
            out.push(0.0);
        }

        debug_assert_eq!(out.len(), self.feature_size());
    }
}
