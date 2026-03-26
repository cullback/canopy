use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum DevCardKind {
    Knight = 0,
    VictoryPoint = 1,
    RoadBuilding = 2,
    YearOfPlenty = 3,
    Monopoly = 4,
}

impl DevCardKind {
    pub const ALL: [DevCardKind; 5] = [
        DevCardKind::Knight,
        DevCardKind::VictoryPoint,
        DevCardKind::RoadBuilding,
        DevCardKind::YearOfPlenty,
        DevCardKind::Monopoly,
    ];
}

/// Fixed-size array indexed by `DevCardKind`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
pub struct DevCardArray(pub [u8; 5]);

impl Index<DevCardKind> for DevCardArray {
    type Output = u8;
    fn index(&self, k: DevCardKind) -> &u8 {
        &self.0[k as usize]
    }
}

impl IndexMut<DevCardKind> for DevCardArray {
    fn index_mut(&mut self, k: DevCardKind) -> &mut u8 {
        &mut self.0[k as usize]
    }
}

/// Development card deck tracking only the total number of remaining cards.
///
/// Per-type distributions are derived from visible information (hands + played
/// piles) via `GameState::unknown_dev_pool()` when needed. This avoids the
/// impossible task of tracking per-type counts when opponent draws are hidden.
#[derive(Clone, Copy, Debug)]
pub struct DevCardDeck {
    pub total: u8,
}

impl DevCardDeck {
    /// Standard deck: 14 knights + 5 VP + 2 road building + 2 year of plenty + 2 monopoly = 25.
    pub fn new() -> Self {
        Self { total: 25 }
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }
}
