/// Axial hex coordinate. Cube coordinate s = -q - r when needed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Hex {
    pub q: i8,
    pub r: i8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Direction {
    East = 0,
    Southeast = 1,
    Southwest = 2,
    West = 3,
    Northwest = 4,
    Northeast = 5,
}

/// Unit vectors in axial (q, r) for each direction.
/// Derived from cube vectors: E=(1,-1,0), SE=(0,-1,1), SW=(-1,0,1),
/// W=(-1,1,0), NW=(0,1,-1), NE=(1,0,-1) with q=x, r=z.
const DIRECTION_VECTORS: [(i8, i8); 6] = [
    (1, 0),  // East
    (0, 1),  // Southeast
    (-1, 1), // Southwest
    (-1, 0), // West
    (0, -1), // Northwest
    (1, -1), // Northeast
];

const SQRT3: f64 = 1.732_050_807_568_877_2;

impl Hex {
    pub const fn new(q: i8, r: i8) -> Self {
        Self { q, r }
    }

    pub const fn neighbor(self, dir: Direction) -> Self {
        let (dq, dr) = DIRECTION_VECTORS[dir as usize];
        Self {
            q: self.q + dq,
            r: self.r + dr,
        }
    }

    /// Convert axial coordinates to pixel center (pointy-top layout).
    pub fn pixel_center(self, size: f64) -> (f64, f64) {
        let x = size * SQRT3 * (self.q as f64 + self.r as f64 / 2.0);
        let y = size * 1.5 * self.r as f64;
        (x, y)
    }
}

/// Pixel offset from hex center for a corner (pointy-top, 0=N clockwise).
pub fn corner_pixel_offset(corner: usize, size: f64) -> (f64, f64) {
    let w = size * SQRT3 / 2.0;
    match corner {
        0 => (0.0, -size),
        1 => (w, -size / 2.0),
        2 => (w, size / 2.0),
        3 => (0.0, size),
        4 => (-w, size / 2.0),
        5 => (-w, -size / 2.0),
        _ => unreachable!(),
    }
}

impl Direction {
    /// The opposite direction.
    pub const fn opposite(self) -> Self {
        match self {
            Self::East => Self::West,
            Self::Southeast => Self::Northwest,
            Self::Southwest => Self::Northeast,
            Self::West => Self::East,
            Self::Northwest => Self::Southeast,
            Self::Northeast => Self::Southwest,
        }
    }
}
