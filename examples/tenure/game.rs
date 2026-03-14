/// Spencer's Attacker-Defender (Tenure) Game.
///
/// An attacker partitions pieces and a defender destroys one partition.
/// The attacker uses micro-actions to build partitions one piece at a time,
/// keeping the action space linear (K+1) instead of exponential.

pub const K: usize = 10;
pub const DONE: usize = K;
pub const DESTROY_A: usize = K + 1;
pub const DESTROY_B: usize = K + 2;
pub const NUM_ACTIONS: usize = K + 3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Attacker,
    Defender,
}

#[derive(Clone, Copy, Debug)]
pub struct TenureGame {
    pub board: [u8; K],
    pub partition: [u8; K],
    pub phase: Phase,
    pub score: u8,
    pub initial_value: f32,
}

impl TenureGame {
    /// Create a new game with `n` pieces at the bottom level (K-1).
    pub fn new(n: u8) -> Self {
        let mut board = [0u8; K];
        board[K - 1] = n;
        Self::with_board(board)
    }

    /// Create a new game with a custom initial board configuration.
    pub fn with_board(board: [u8; K]) -> Self {
        let initial_value = optimal_value(&board);
        Self {
            board,
            partition: [0; K],
            phase: Phase::Attacker,
            score: 0,
            initial_value,
        }
    }

    /// Generate a random board with a normally-distributed target potential.
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        // Box-Muller: N(2.1, 0.5)
        let u1 = 1.0 - rng.f64(); // (0, 1] to avoid ln(0)
        let u2 = rng.f64();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let min = 1.0 / (1u32 << K) as f64;
        let potential = (2.1 + 0.05 * z).max(min);
        Self::random_with_potential(potential as f32, rng)
    }

    /// Generate a random board targeting a specific total optimal value.
    /// Pieces are placed with a geometric distribution favoring levels
    /// closer to tenure (higher-value pieces).
    pub fn random_with_potential(mut potential: f32, rng: &mut fastrand::Rng) -> Self {
        let mut board = [0u8; K];
        let min_value = 1.0 / (1u32 << K) as f32;
        while potential >= min_value {
            // Geometric: level k with probability 1/2^(k+1)
            let mut level = 0;
            while level < K - 1 && rng.bool() {
                level += 1;
            }
            let value = 1.0 / (1u32 << (level + 1)) as f32;
            if value <= potential {
                board[level] += 1;
                potential -= value;
            }
        }
        Self::with_board(board)
    }

    /// True when no pieces remain on the board or partition.
    pub fn is_terminal(&self) -> bool {
        self.board
            .iter()
            .chain(self.partition.iter())
            .all(|&x| x == 0)
    }

    /// Terminal reward from the attacker's (P1) perspective.
    /// The threshold is floor(v*), so score > floor(v*) = attacker wins,
    /// score == floor(v*) = draw, score < floor(v*) = defender wins.
    pub fn terminal_reward(&self) -> f32 {
        let threshold = self.initial_value.floor() as u8;
        if self.score > threshold {
            1.0
        } else if self.score < threshold {
            -1.0
        } else {
            0.0
        }
    }

    /// Attacker micro-action: move one piece from board to partition at the given level.
    pub fn attacker_move(&mut self, level: usize) {
        debug_assert_eq!(self.phase, Phase::Attacker);
        debug_assert!(level < K);
        debug_assert!(self.board[level] > 0);
        self.board[level] -= 1;
        self.partition[level] += 1;
    }

    /// Attacker finishes building the partition.
    pub fn attacker_done(&mut self) {
        debug_assert_eq!(self.phase, Phase::Attacker);
        self.phase = Phase::Defender;
    }

    /// Defender destroys one partition. If `destroy_b`, destroy the partition
    /// (keep board). Otherwise, destroy the board (keep partition).
    /// Scores level-0 pieces from the survivor, shifts remaining pieces up.
    pub fn defender_choose(&mut self, destroy_b: bool) {
        debug_assert_eq!(self.phase, Phase::Defender);
        let survivor = if destroy_b {
            self.board
        } else {
            self.partition
        };
        self.score += survivor[0];
        for i in 0..K - 1 {
            self.board[i] = survivor[i + 1];
        }
        self.board[K - 1] = 0;
        self.partition = [0; K];
        self.phase = Phase::Attacker;
    }
}

/// Compute the optimal (game-theoretic) value of a board configuration.
/// v*(i) = 1/2^(i+1), v*(S) = sum(S[i] * v*(i)).
pub fn optimal_value(board: &[u8; K]) -> f32 {
    (0..K)
        .map(|i| board[i] as f32 / (1u32 << (i + 1)) as f32)
        .sum()
}
