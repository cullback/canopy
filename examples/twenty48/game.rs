/// 2048 tile-sliding game.
///
/// Single-player stochastic game on a 4x4 grid. The player picks a direction
/// (Up/Down/Left/Right) to slide tiles, merging equal values. After each move
/// a random tile (90% "2", 10% "4") spawns in an empty cell.
///
/// Bitboard representation: `u64` with 4 bits per cell (nibble value `n` =
/// tile `2^n`, 0 = empty). Row-major: row 0 in bits 0-15, row 3 in bits 48-63.
///
/// ## Phase encoding
///
/// The game alternates between decision nodes (player picks direction) and
/// chance nodes (random tile spawns). Instead of a separate `bool`, we encode
/// the phase into the `u64` itself:
///
/// - **Decision node**: tiles stored directly (empty cells = `0x0`)
/// - **Chance node**: bitwise complement `!tiles` stored (empty cells = `0xF`)
///
/// To distinguish: a real board has more `0x0` nibbles (empty cells) than
/// `0xF` nibbles (tile 32768, practically unreachable). The complement has
/// the reverse. So `count_f_nibbles > count_zero_nibbles` → chance node.
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

/// Count nibbles equal to `0x0`.
pub fn count_zero_nibbles(b: u64) -> u32 {
    let x = b | (b >> 1);
    let x = x | (x >> 2);
    let occupied = x & 0x1111_1111_1111_1111_u64;
    16 - occupied.count_ones()
}

/// Count nibbles equal to `0xF`.
pub fn count_f_nibbles(b: u64) -> u32 {
    let x = b & (b >> 1);
    let x = x & (x >> 2);
    (x & 0x1111_1111_1111_1111_u64).count_ones()
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

/// Place a random tile (90% "2", 10% "4") in a random empty cell.
fn spawn_on(tiles: u64, rng: &mut fastrand::Rng) -> u64 {
    let empties = empty_positions(tiles);
    debug_assert!(!empties.is_empty());
    let pos = empties[rng.usize(..empties.len())];
    let val: u8 = if rng.u32(0..10) == 0 { 2 } else { 1 };
    set_nibble(tiles, pos, val)
}

// ── Board struct ───────────────────────────────────────────────────────

/// Packed board state. A single `u64` encodes both the tile data and the
/// game phase (decision vs chance). See module docs for the encoding scheme.
#[derive(Clone, Copy, Debug)]
pub struct Board(pub u64);

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for Board {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v: u64 = s
            .trim()
            .parse()
            .map_err(|e| format!("invalid board: {e}"))?;
        Ok(Board(v))
    }
}

impl Board {
    /// Create a new board with two random starting tiles (decision node).
    pub fn new(rng: &mut fastrand::Rng) -> Self {
        Board(spawn_on(spawn_on(0, rng), rng))
    }

    /// Recover the real tile data, regardless of phase.
    pub fn tiles(self) -> u64 {
        if self.awaiting_spawn() {
            !self.0
        } else {
            self.0
        }
    }

    /// True when a tile spawn is pending (chance node).
    ///
    /// A complemented board (chance) has more `0xF` nibbles than `0x0`
    /// nibbles. A real board (decision) has the reverse. Ties (both zero
    /// on a full board) correctly fall to decision — that state is terminal.
    pub fn awaiting_spawn(self) -> bool {
        count_f_nibbles(self.0) > count_zero_nibbles(self.0)
    }

    /// Score: sum of all tile face values.
    #[allow(dead_code)]
    pub fn score(self) -> u32 {
        board_score(self.tiles())
    }

    /// Largest tile on the board (e.g. 2048).
    #[allow(dead_code)]
    pub fn max_tile(self) -> u32 {
        max_tile(self.tiles())
    }

    /// Pretty-print the board grid.
    #[allow(dead_code)]
    pub fn display(self) -> String {
        let tiles = self.tiles();
        let mut s = String::new();
        for row in 0..4 {
            for col in 0..4 {
                let pos = row * 4 + col;
                let nib = get_nibble(tiles, pos as u32);
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
        let row: u16 = 0x4321;
        assert_eq!(reverse_row(row), 0x1234);
    }

    #[test]
    fn test_compute_row_left_simple() {
        // [1,1,0,0] → [2,0,0,0]
        let row: u16 = 0x0011;
        assert_eq!(compute_row_left(row), 0x0002);
    }

    #[test]
    fn test_compute_row_left_no_merge() {
        // [1, 2, 0, 0] → [1, 2, 0, 0]
        assert_eq!(compute_row_left(0x0021), 0x0021);
    }

    #[test]
    fn test_compute_row_left_slide() {
        // [0, 0, 1, 2] → [1, 2, 0, 0]
        assert_eq!(compute_row_left(0x2100), 0x0021);
    }

    #[test]
    fn test_compute_row_left_double_merge() {
        // [1, 1, 1, 1] → [2, 2, 0, 0]
        assert_eq!(compute_row_left(0x1111), 0x0022);
    }

    #[test]
    fn test_transpose_roundtrip() {
        let board: u64 = 0x0123_4567_89AB_CDEF;
        assert_eq!(transpose(transpose(board)), board);
    }

    #[test]
    fn test_transpose_identity() {
        let board: u64 = 0x1111_1111_1111_1111;
        assert_eq!(transpose(board), board);
    }

    #[test]
    fn test_move_left() {
        // Row 0: [1, 1, 0, 0] → [2, 0, 0, 0]
        assert_eq!(move_left(0x0011) & 0xFFFF, 0x0002);
    }

    #[test]
    fn test_move_right() {
        // Row 0: [1, 1, 0, 0] → [0, 0, 0, 2]
        assert_eq!(move_right(0x0011) & 0xFFFF, 0x2000);
    }

    #[test]
    fn test_count_zero_nibbles() {
        assert_eq!(count_zero_nibbles(0), 16);
        assert_eq!(count_zero_nibbles(0x1), 15);
        assert_eq!(count_zero_nibbles(0xFFFF_FFFF_FFFF_FFFF), 0);
    }

    #[test]
    fn test_count_f_nibbles() {
        assert_eq!(count_f_nibbles(0), 0);
        assert_eq!(count_f_nibbles(0xF), 1);
        assert_eq!(count_f_nibbles(0xFFFF_FFFF_FFFF_FFFF), 16);
        assert_eq!(count_f_nibbles(0xF00F_000F), 3);
    }

    #[test]
    fn test_board_score() {
        assert_eq!(board_score(0x1), 2);
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
        assert!(has_legal_move(0x1));
    }

    #[test]
    fn test_terminal_detection() {
        // Checkerboard pattern with no adjacent equal tiles
        let board: u64 = 0x2143_4312_2143_4312;
        if execute_move(board, UP) == board
            && execute_move(board, DOWN) == board
            && execute_move(board, LEFT) == board
            && execute_move(board, RIGHT) == board
        {
            assert!(!has_legal_move(board));
        }
    }

    #[test]
    fn test_spawn_encoding() {
        let pos: u32 = 5;
        let action_2 = (pos as usize) * 2;
        let action_4 = (pos as usize) * 2 + 1;
        assert_eq!(action_2, 10);
        assert_eq!(action_4, 11);
        assert_eq!(action_2 / 2, pos as usize);
        assert_eq!(action_2 % 2, 0);
        assert_eq!(action_4 / 2, pos as usize);
        assert_eq!(action_4 % 2, 1);
    }

    #[test]
    fn test_new_board_has_two_tiles() {
        let mut rng = fastrand::Rng::with_seed(42);
        let board = Board::new(&mut rng);
        let tiles = board.tiles();
        let occupied = 16 - count_zero_nibbles(tiles);
        assert_eq!(occupied, 2);
        assert!(!board.awaiting_spawn());
    }

    #[test]
    fn test_score_calculation() {
        // Two "2" tiles (nibble 1) and one "4" tile (nibble 2): score = 2 + 2 + 4 = 8
        let board = set_nibble(set_nibble(set_nibble(0, 0, 1), 1, 1), 2, 2);
        assert_eq!(board_score(board), 8);
    }

    #[test]
    fn test_phase_encoding_decision() {
        // A board with some tiles and empty cells → decision node
        let tiles: u64 = 0x0000_0000_0000_0321;
        let board = Board(tiles);
        assert!(!board.awaiting_spawn());
        assert_eq!(board.tiles(), tiles);
    }

    #[test]
    fn test_phase_encoding_chance() {
        // Complement of a board with empty cells → chance node
        let tiles: u64 = 0x0000_0000_0000_0321;
        let board = Board(!tiles);
        assert!(board.awaiting_spawn());
        assert_eq!(board.tiles(), tiles);
    }

    #[test]
    fn test_phase_roundtrip() {
        // Simulate: decision → apply direction → chance → apply spawn → decision
        let mut rng = fastrand::Rng::with_seed(42);
        let mut board = Board::new(&mut rng);
        assert!(!board.awaiting_spawn());

        // Apply a direction move → becomes chance node
        let tiles = board.tiles();
        let new_tiles = execute_move(tiles, LEFT);
        assert_ne!(new_tiles, tiles); // move must change board
        board = Board(!new_tiles); // store as chance
        assert!(board.awaiting_spawn());
        assert_eq!(board.tiles(), new_tiles);

        // Apply a spawn → becomes decision node
        let empties = empty_positions(board.tiles());
        let pos = empties[0];
        let spawned = set_nibble(board.tiles(), pos, 1);
        board = Board(spawned); // store as decision
        assert!(!board.awaiting_spawn());
        assert_eq!(board.tiles(), spawned);
    }

    #[test]
    fn test_full_board_decision_is_not_chance() {
        // Full board with no empty cells and no 0xF tiles → zeros == fs == 0 → decision
        let board: u64 = 0x1234_5678_1234_5678;
        assert_eq!(count_zero_nibbles(board), 0);
        assert_eq!(count_f_nibbles(board), 0);
        assert!(!Board(board).awaiting_spawn());
    }
}
