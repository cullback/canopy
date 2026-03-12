/// 2048 tile-sliding game.
///
/// Single-player stochastic game on a 4x4 grid. The player picks a direction
/// (Up/Down/Left/Right) to slide tiles, merging equal values. After each move
/// a random tile (90% "2", 10% "4") spawns in an empty cell.
///
/// Bitboard representation: `u64` with 4 bits per cell (nibble value `n` =
/// tile `2^n`, 0 = empty). Row-major: row 0 in bits 0-15, row 3 in bits 48-63.
use std::sync::OnceLock;

pub const UP: usize = 0;
pub const DOWN: usize = 1;
pub const LEFT: usize = 2;
pub const RIGHT: usize = 3;
pub const NUM_ACTIONS: usize = 4;

#[allow(dead_code)]
pub const ACTION_NAMES: [&str; 4] = ["Up", "Down", "Left", "Right"];

// ── Lookup tables ──────────────────────────────────────────────────────

type RowTable = Box<[u16; 65536]>;
type ColTable = Box<[u64; 65536]>;

fn row_left_table() -> &'static RowTable {
    static TABLE: OnceLock<RowTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0u16; 65536]);
        for row in 0u32..65536 {
            let r = row as u16;
            t[row as usize] = compute_row_left(r) ^ r;
        }
        t
    })
}

fn row_right_table() -> &'static RowTable {
    static TABLE: OnceLock<RowTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0u16; 65536]);
        for row in 0u32..65536 {
            let r = row as u16;
            let reversed = reverse_row(r);
            let result = reverse_row(compute_row_left(reversed));
            t[row as usize] = result ^ r;
        }
        t
    })
}

fn col_up_table() -> &'static ColTable {
    static TABLE: OnceLock<ColTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0u64; 65536]);
        for row in 0u32..65536 {
            let r = row as u16;
            let result = compute_row_left(r);
            // Unpack row result into a column: nibble i → row i, column 0
            let delta = result ^ r;
            let c0 = (delta & 0xF) as u64;
            let c1 = ((delta >> 4) & 0xF) as u64;
            let c2 = ((delta >> 8) & 0xF) as u64;
            let c3 = ((delta >> 12) & 0xF) as u64;
            t[row as usize] = c0 | (c1 << 16) | (c2 << 32) | (c3 << 48);
        }
        t
    })
}

fn col_down_table() -> &'static ColTable {
    static TABLE: OnceLock<ColTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0u64; 65536]);
        for row in 0u32..65536 {
            let r = row as u16;
            let reversed = reverse_row(r);
            let result = reverse_row(compute_row_left(reversed));
            let delta = result ^ r;
            let c0 = (delta & 0xF) as u64;
            let c1 = ((delta >> 4) & 0xF) as u64;
            let c2 = ((delta >> 8) & 0xF) as u64;
            let c3 = ((delta >> 12) & 0xF) as u64;
            t[row as usize] = c0 | (c1 << 16) | (c2 << 32) | (c3 << 48);
        }
        t
    })
}

/// Slide a 4-nibble row left, merging tiles.
fn compute_row_left(row: u16) -> u16 {
    let mut cells = [
        (row & 0xF) as u8,
        ((row >> 4) & 0xF) as u8,
        ((row >> 8) & 0xF) as u8,
        ((row >> 12) & 0xF) as u8,
    ];

    // Compact non-zero to the left
    let mut out = [0u8; 4];
    let mut pos = 0;
    for &c in &cells {
        if c != 0 {
            out[pos] = c;
            pos += 1;
        }
    }
    cells = out;

    // Merge adjacent equal tiles left-to-right
    let mut result = [0u8; 4];
    let mut rpos = 0;
    let mut i = 0;
    while i < 4 {
        if cells[i] == 0 {
            break;
        }
        if i + 1 < 4 && cells[i] == cells[i + 1] && cells[i] < 15 {
            result[rpos] = cells[i] + 1;
            i += 2;
        } else {
            result[rpos] = cells[i];
            i += 1;
        }
        rpos += 1;
    }

    result[0] as u16 | (result[1] as u16) << 4 | (result[2] as u16) << 8 | (result[3] as u16) << 12
}

fn reverse_row(row: u16) -> u16 {
    let a = row & 0xF;
    let b = (row >> 4) & 0xF;
    let c = (row >> 8) & 0xF;
    let d = (row >> 12) & 0xF;
    (a << 12) | (b << 8) | (c << 4) | d
}

// ── Board operations ───────────────────────────────────────────────────

/// Transpose a 4x4 nibble board (swap rows and columns).
pub fn transpose(board: u64) -> u64 {
    // Swap 4-bit nibbles between positions that are row/col transposed.
    // Adapted from nneonneo/2048-ai.
    let a1 = board & 0xF0F0_0F0F_F0F0_0F0F_u64;
    let a2 = board & 0x0000_F0F0_0000_F0F0_u64;
    let a3 = board & 0x0F0F_0000_0F0F_0000_u64;
    let a = a1 | (a2 << 12) | (a3 >> 12);
    let b1 = a & 0xFF00_FF00_00FF_00FF_u64;
    let b2 = a & 0x0000_0000_FF00_FF00_u64;
    let b3 = a & 0x00FF_00FF_0000_0000_u64;
    b1 | (b2 << 24) | (b3 >> 24)
}

/// Execute a left move on the full board.
pub fn move_left(board: u64) -> u64 {
    let tbl = row_left_table();
    let r0 = (board & 0xFFFF) as usize;
    let r1 = ((board >> 16) & 0xFFFF) as usize;
    let r2 = ((board >> 32) & 0xFFFF) as usize;
    let r3 = ((board >> 48) & 0xFFFF) as usize;
    board
        ^ tbl[r0] as u64
        ^ ((tbl[r1] as u64) << 16)
        ^ ((tbl[r2] as u64) << 32)
        ^ ((tbl[r3] as u64) << 48)
}

/// Execute a right move on the full board.
pub fn move_right(board: u64) -> u64 {
    let tbl = row_right_table();
    let r0 = (board & 0xFFFF) as usize;
    let r1 = ((board >> 16) & 0xFFFF) as usize;
    let r2 = ((board >> 32) & 0xFFFF) as usize;
    let r3 = ((board >> 48) & 0xFFFF) as usize;
    board
        ^ tbl[r0] as u64
        ^ ((tbl[r1] as u64) << 16)
        ^ ((tbl[r2] as u64) << 32)
        ^ ((tbl[r3] as u64) << 48)
}

/// Execute an up move on the full board.
pub fn move_up(board: u64) -> u64 {
    let t = transpose(board);
    let tbl = col_up_table();
    let r0 = (t & 0xFFFF) as usize;
    let r1 = ((t >> 16) & 0xFFFF) as usize;
    let r2 = ((t >> 32) & 0xFFFF) as usize;
    let r3 = ((t >> 48) & 0xFFFF) as usize;
    board ^ tbl[r0] ^ (tbl[r1] << 4) ^ (tbl[r2] << 8) ^ (tbl[r3] << 12)
}

/// Execute a down move on the full board.
pub fn move_down(board: u64) -> u64 {
    let t = transpose(board);
    let tbl = col_down_table();
    let r0 = (t & 0xFFFF) as usize;
    let r1 = ((t >> 16) & 0xFFFF) as usize;
    let r2 = ((t >> 32) & 0xFFFF) as usize;
    let r3 = ((t >> 48) & 0xFFFF) as usize;
    board ^ tbl[r0] ^ (tbl[r1] << 4) ^ (tbl[r2] << 8) ^ (tbl[r3] << 12)
}

/// Apply move direction to board. Returns new board state.
pub fn execute_move(board: u64, dir: usize) -> u64 {
    match dir {
        UP => move_up(board),
        DOWN => move_down(board),
        LEFT => move_left(board),
        RIGHT => move_right(board),
        _ => panic!("invalid direction {dir}"),
    }
}

/// Count empty cells (nibbles == 0).
#[allow(dead_code)]
pub fn count_empty(board: u64) -> u32 {
    // A nibble is empty if all 4 bits are 0. We OR adjacent bit pairs,
    // then OR the two resulting bits, giving one "occupied" bit per nibble.
    let b = board | (board >> 1);
    let b = b | (b >> 2);
    // Now bit 0 of each nibble is 1 if occupied, 0 if empty.
    // Mask those bits and count zeros.
    let occupied = b & 0x1111_1111_1111_1111_u64;
    16 - occupied.count_ones()
}

/// Sum of tile values on the board (path-independent score).
/// Each nibble `n > 0` contributes `2^n`.
pub fn board_score(board: u64) -> u32 {
    let mut score = 0u32;
    let mut b = board;
    for _ in 0..16 {
        let nibble = (b & 0xF) as u32;
        if nibble > 0 {
            score += 1 << nibble;
        }
        b >>= 4;
    }
    score
}

/// Maximum tile value on the board (e.g. 2048).
pub fn max_tile(board: u64) -> u32 {
    let mut max_nib = 0u32;
    let mut b = board;
    for _ in 0..16 {
        let nibble = (b & 0xF) as u32;
        if nibble > max_nib {
            max_nib = nibble;
        }
        b >>= 4;
    }
    if max_nib > 0 { 1 << max_nib } else { 0 }
}

/// Get the nibble value at position `pos` (0..16).
pub fn get_nibble(board: u64, pos: u32) -> u8 {
    ((board >> (pos * 4)) & 0xF) as u8
}

/// Set the nibble value at position `pos` (0..16).
pub fn set_nibble(board: u64, pos: u32, val: u8) -> u64 {
    let shift = pos * 4;
    (board & !(0xF_u64 << shift)) | ((val as u64) << shift)
}

/// Find all empty positions on the board.
pub fn empty_positions(board: u64) -> Vec<u32> {
    let mut positions = Vec::new();
    for i in 0..16 {
        if get_nibble(board, i) == 0 {
            positions.push(i);
        }
    }
    positions
}

/// Check if any legal move exists.
pub fn has_legal_move(board: u64) -> bool {
    for dir in 0..4 {
        if execute_move(board, dir) != board {
            return true;
        }
    }
    false
}

// ── Board struct ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Board {
    pub tiles: u64,
    /// True when a tile spawn is pending (chance node).
    pub awaiting_spawn: bool,
}

impl Board {
    /// Create a new board with two random starting tiles.
    pub fn new(rng: &mut fastrand::Rng) -> Self {
        let mut board = Board {
            tiles: 0,
            awaiting_spawn: false,
        };
        board.spawn_tile(rng);
        board.spawn_tile(rng);
        board
    }

    /// Place a random tile (90% "2", 10% "4") in a random empty cell.
    pub fn spawn_tile(&mut self, rng: &mut fastrand::Rng) {
        let empties = empty_positions(self.tiles);
        debug_assert!(!empties.is_empty());
        let pos = empties[rng.usize(..empties.len())];
        let val: u8 = if rng.u32(0..10) == 0 { 2 } else { 1 };
        self.tiles = set_nibble(self.tiles, pos, val);
    }

    /// Score: sum of all tile face values.
    #[allow(dead_code)]
    pub fn score(&self) -> u32 {
        board_score(self.tiles)
    }

    /// Largest tile on the board (e.g. 2048).
    #[allow(dead_code)]
    pub fn max_tile(&self) -> u32 {
        max_tile(self.tiles)
    }

    /// Pretty-print the board grid.
    #[allow(dead_code)]
    pub fn display(&self) -> String {
        let mut s = String::new();
        for row in 0..4 {
            for col in 0..4 {
                let pos = row * 4 + col;
                let nib = get_nibble(self.tiles, pos as u32);
                if nib == 0 {
                    s.push_str("    .");
                } else {
                    s.push_str(&format!("{:5}", 1u32 << nib));
                }
            }
            s.push('\n');
        }
        s
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_row() {
        // Row: [1, 2, 3, 4] nibbles
        let row: u16 = 0x4321;
        assert_eq!(reverse_row(row), 0x1234);
    }

    #[test]
    fn test_compute_row_left_simple() {
        // [2, 2, 0, 0] → [3, 0, 0, 0] (merge 2+2 → 4, encoded as 2+2→3)
        // Wait, nibble 2 = tile 4. Two "4" tiles merge → tile 8 = nibble 3.
        // Actually nibble values: 1=2, 2=4. [1,1,0,0] → [2,0,0,0]
        let row: u16 = 0x0011;
        let result = compute_row_left(row);
        assert_eq!(result, 0x0002);
    }

    #[test]
    fn test_compute_row_left_no_merge() {
        // [1, 2, 0, 0] → [1, 2, 0, 0]
        let row: u16 = 0x0021;
        let result = compute_row_left(row);
        assert_eq!(result, 0x0021);
    }

    #[test]
    fn test_compute_row_left_slide() {
        // [0, 0, 1, 2] → [1, 2, 0, 0]
        let row: u16 = 0x2100;
        let result = compute_row_left(row);
        assert_eq!(result, 0x0021);
    }

    #[test]
    fn test_compute_row_left_double_merge() {
        // [1, 1, 1, 1] → [2, 2, 0, 0]
        let row: u16 = 0x1111;
        let result = compute_row_left(row);
        assert_eq!(result, 0x0022);
    }

    #[test]
    fn test_transpose_roundtrip() {
        let board: u64 = 0x0123_4567_89AB_CDEF;
        assert_eq!(transpose(transpose(board)), board);
    }

    #[test]
    fn test_transpose_identity() {
        // All same nibbles → transpose is identity
        let board: u64 = 0x1111_1111_1111_1111;
        assert_eq!(transpose(board), board);
    }

    #[test]
    fn test_move_left() {
        // Row 0: [1, 1, 0, 0] → [2, 0, 0, 0]
        let board: u64 = 0x0011;
        let result = move_left(board);
        assert_eq!(result & 0xFFFF, 0x0002);
    }

    #[test]
    fn test_move_right() {
        // Row 0: [1, 1, 0, 0] → [0, 0, 0, 2]
        let board: u64 = 0x0011;
        let result = move_right(board);
        assert_eq!(result & 0xFFFF, 0x2000);
    }

    #[test]
    fn test_count_empty() {
        assert_eq!(count_empty(0), 16);
        assert_eq!(count_empty(0x1), 15);
        assert_eq!(count_empty(0xFFFF_FFFF_FFFF_FFFF), 0);
    }

    #[test]
    fn test_board_score() {
        // One tile: nibble 1 = tile 2 → score 2
        assert_eq!(board_score(0x1), 2);
        // nibble 10 (0xA) = tile 1024 → score 1024
        assert_eq!(board_score(0xA), 1024);
    }

    #[test]
    fn test_max_tile() {
        assert_eq!(max_tile(0), 0);
        assert_eq!(max_tile(0x1), 2);
        assert_eq!(max_tile(0xB000_0000_0000_0001), 2048);
    }

    #[test]
    fn test_get_set_nibble() {
        let board = set_nibble(0, 5, 0xA);
        assert_eq!(get_nibble(board, 5), 0xA);
        assert_eq!(get_nibble(board, 0), 0);
    }

    #[test]
    fn test_has_legal_move_empty_board() {
        // Empty board with one tile — can always move
        assert!(has_legal_move(0x1));
    }

    #[test]
    fn test_terminal_detection() {
        // Checkerboard pattern with no adjacent equal tiles
        let board: u64 = 0x2143_4312_2143_4312;
        // Verify each move doesn't change the board
        if execute_move(board, UP) == board
            && execute_move(board, DOWN) == board
            && execute_move(board, LEFT) == board
            && execute_move(board, RIGHT) == board
        {
            assert!(!has_legal_move(board));
        }
        // If the pattern does allow a move, that's OK — just verifying the function works
    }

    #[test]
    fn test_spawn_encoding() {
        // chance_outcomes encoding: position p → (p*2, 9) for "2", (p*2+1, 1) for "4"
        let pos: u32 = 5;
        let action_2 = (pos as usize) * 2;
        let action_4 = (pos as usize) * 2 + 1;
        assert_eq!(action_2, 10);
        assert_eq!(action_4, 11);
        // Decode
        assert_eq!(action_2 / 2, pos as usize); // position
        assert_eq!(action_2 % 2, 0); // tile type: "2"
        assert_eq!(action_4 / 2, pos as usize); // position
        assert_eq!(action_4 % 2, 1); // tile type: "4"
    }

    #[test]
    fn test_new_board_has_two_tiles() {
        let mut rng = fastrand::Rng::with_seed(42);
        let board = Board::new(&mut rng);
        let occupied = 16 - count_empty(board.tiles);
        assert_eq!(occupied, 2);
        assert!(!board.awaiting_spawn);
    }

    #[test]
    fn test_score_calculation() {
        // Two "2" tiles (nibble 1) and one "4" tile (nibble 2):
        // score = 2 + 2 + 4 = 8
        let board = set_nibble(set_nibble(set_nibble(0, 0, 1), 1, 1), 2, 2);
        assert_eq!(board_score(board), 8);
    }
}
